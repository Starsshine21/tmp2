import os

import tyro


def main():
    for proxy_var in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
        "no_proxy",
        "NO_PROXY",
    ):
        os.environ.pop(proxy_var, None)

    from dexjoco_openpi_client.eval_dexjoco_openpi import main as evaluate_dexjoco_openpi

    tyro.cli(evaluate_dexjoco_openpi)


if __name__ == "__main__":
    main()
