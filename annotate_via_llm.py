#!/usr/bin/env python
'''
Annotate image clusters with a vision-language model.

annotate_via_kym.py can only label a cluster whose medoid is within a few bits
of an image already in the Know Your Meme pHash list. This script instead shows
the medoid (and a few other members) of each cluster to Claude and asks what the
meme is, so clusters with no KYM match still get a name and a description.

Input:  the clustering output of clustering_phashes.py (default clustering_output.txt)
Output: a json file {cluster_no: {name, description, visible_text, confidence, images_shown}}

Credentials: ANTHROPIC_API_KEY by default, or --bedrock-region to go through
Amazon Bedrock with your AWS credentials (pip install "anthropic[bedrock]").
'''
import argparse
import ast
import base64
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import anthropic
from PIL import Image

DEFAULT_MODEL = 'claude-opus-5-5'
FALLBACK_BETA = 'server-side-fallback-2026-07-01'
MAX_SIDE = 1568  # longest image side sent to the model; larger images are scaled down

SYSTEM = '''You annotate clusters of near-duplicate images for a research pipeline that studies how \
memes spread between web communities. Each request contains a few images from one cluster; the first \
is the cluster's most representative image.

Identify the meme or image template the cluster is built on. Use the name it is commonly known by \
(for example the title it would have on Know Your Meme). If it is not a recognisable meme, leave the \
name empty and just describe the image.

Some images come from fringe communities and can be offensive. Describe what is shown neutrally and \
factually, including when a meme is used to express hate. Do not identify real private individuals.'''

SCHEMA = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string', 'description': 'Common name of the meme or template; empty string if not a recognisable meme.'},
        'description': {'type': 'string', 'description': 'One or two sentences on what the images show and how the meme is typically used.'},
        'visible_text': {'type': 'string', 'description': 'Text visible in the representative image; empty string if none.'},
        'confidence': {'type': 'string', 'enum': ['low', 'medium', 'high'], 'description': 'Confidence in the name.'},
    },
    'required': ['name', 'description', 'visible_text', 'confidence'],
    'additionalProperties': False,
}


def process_clusters_file(clusters_file):
    '''Return {cluster_no: [medoid_path, other image paths...]}, skipping the noise cluster.'''
    clusters = {}
    with open(clusters_file) as fd:
        for line in fd:
            if not line.strip():
                continue
            dic = ast.literal_eval(line)
            cluster_no = int(dic['cluster_no'])
            if cluster_no == -1:
                continue
            medoid = dic['medroid_path']
            clusters[cluster_no] = [medoid] + [im for im in dic['images'] if im != medoid]
    return clusters


def image_block(path):
    '''Load an image, scale it down if needed, and return it as an API image block.'''
    img = Image.open(path)
    img.seek(0)  # first frame of animated images
    has_alpha = img.mode in ('RGBA', 'LA', 'P')
    img = img.convert('RGBA' if has_alpha else 'RGB')
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    if has_alpha:
        img.save(buf, format='PNG')
        media_type = 'image/png'
    else:
        img.save(buf, format='JPEG', quality=90)
        media_type = 'image/jpeg'
    data = base64.standard_b64encode(buf.getvalue()).decode('utf-8')
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': media_type, 'data': data}}


def make_client(bedrock_region):
    if bedrock_region:
        from anthropic import AnthropicBedrockMantle
        return AnthropicBedrockMantle(aws_region=bedrock_region)
    return anthropic.Anthropic()


def annotate_cluster(client, model, bedrock, cluster_no, images, samples):
    shown = [im for im in images if os.path.isfile(im)][:samples]
    if not shown:
        return {'error': 'no image files found for this cluster'}
    content = [image_block(im) for im in shown]
    content.append({'type': 'text', 'text': 'Annotate this cluster (%d images in total, %d shown).' % (len(images), len(shown))})
    request = dict(
        model=model,
        max_tokens=4096,
        system=SYSTEM,
        output_config={'effort': 'low', 'format': {'type': 'json_schema', 'schema': SCHEMA}},
        messages=[{'role': 'user', 'content': content}],
    )
    try:
        if bedrock:
            response = client.messages.create(**request)
        else:
            # If a safety classifier declines the request, let the API retry it on a
            # fallback model instead of failing. Not available on Bedrock.
            response = client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks='default', **request)
    except anthropic.RateLimitError as e:
        return {'error': 'rate limited after retries: %s' % e.message}
    except anthropic.APIStatusError as e:
        return {'error': 'API error %d: %s' % (e.status_code, e.message)}
    except anthropic.APIConnectionError as e:
        return {'error': 'connection error: %s' % e}
    if response.stop_reason in ('refusal', 'max_tokens'):
        return {'error': 'no annotation (%s)' % response.stop_reason, 'images_shown': shown}
    annotation = json.loads(next(b.text for b in response.content if b.type == 'text'))
    annotation['images_shown'] = shown
    return annotation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate image clusters with a vision-language model.')
    parser.add_argument('-c', '--clustering', default='clustering_output.txt', help='file with the clustering output')
    parser.add_argument('-o', '--output', default='llm_annotations.json', help='file to store the annotations')
    parser.add_argument('-s', '--samples', type=int, default=3, help='images shown to the model per cluster, medoid first')
    parser.add_argument('-m', '--model', default=DEFAULT_MODEL, help='Claude model id')
    parser.add_argument('-w', '--workers', type=int, default=4, help='clusters annotated concurrently')
    parser.add_argument('--bedrock-region', help='call Claude through Amazon Bedrock in this region')
    options = parser.parse_args()

    clusters = process_clusters_file(options.clustering)
    print('[i] annotating %d clusters with %s' % (len(clusters), options.model))
    client = make_client(options.bedrock_region)
    bedrock = options.bedrock_region is not None
    model = options.model
    if bedrock and not model.startswith('anthropic.'):
        model = 'anthropic.' + model

    def run(item):
        cluster_no, images = item
        return cluster_no, annotate_cluster(client, model, bedrock, cluster_no, images, options.samples)

    annotations = {}
    with ThreadPoolExecutor(max_workers=options.workers) as pool:
        for cluster_no, annotation in pool.map(run, sorted(clusters.items())):
            annotations[cluster_no] = annotation
            label = annotation.get('name') or annotation.get('error') or '(not a recognised meme)'
            print('[i] cluster %d: %s' % (cluster_no, label))

    with open(options.output, 'w') as fd:
        json.dump(annotations, fd, indent=2, ensure_ascii=False)
    failed = sum(1 for a in annotations.values() if 'error' in a)
    print('[i] wrote %s (%d annotated, %d failed)' % (options.output, len(annotations) - failed, failed))
    sys.exit(1 if failed else 0)
