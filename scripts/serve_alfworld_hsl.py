"""Existing vLLM OpenAI server plus local synchronous weight-transfer routes."""
from fastapi import Request
from vllm.entrypoints.openai import api_server


async def rpc(request, name, **kwargs):
    results = await api_server.engine_client(request).collective_rpc(name, kwargs=kwargs)
    if len(results) != 1:
        raise ValueError('this launcher uses one tensor-parallel worker')
    return results[0]


@api_server.router.post('/hsl/begin')
async def begin(request: Request):
    payload = await request.json()
    return await rpc(request, 'hsl_begin', version=payload['version'])


@api_server.router.post('/hsl/load_bucket')
async def load_bucket(request: Request):
    payload = await request.json()
    return await rpc(request, 'hsl_load_bucket', **payload)


@api_server.router.post('/hsl/commit')
async def commit(request: Request):
    payload = await request.json()
    result = await rpc(request, 'hsl_commit', **payload)
    # A prefix encoded under old weights cannot be reused after the update.
    await api_server.engine_client(request).reset_prefix_cache()
    return result


@api_server.router.get('/hsl/policy_version')
async def policy_version(request: Request):
    return await rpc(request, 'hsl_policy_version')


@api_server.router.post('/hsl/probe_snapshot')
async def probe_snapshot(request: Request):
    return await rpc(request, 'hsl_probe_snapshot')


@api_server.router.post('/hsl/probe_compare')
async def probe_compare(request: Request):
    return await rpc(request, 'hsl_probe_compare')


if __name__ == '__main__':
    api_server.cli_env_setup()
    parser = api_server.make_arg_parser(api_server.FlexibleArgumentParser())
    args = parser.parse_args()
    args.worker_extension_cls = 'alfworld_hsl_weight_sync.HSLWeightWorker'
    args.host = '127.0.0.1'
    api_server.validate_parsed_serve_args(args)
    api_server.uvloop.run(api_server.run_server(args))
