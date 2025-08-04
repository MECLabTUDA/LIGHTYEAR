from nvflare.app_opt.p2p.types import Config, Node, Network, Neighbor


def camelyon17_config():
    config = Config(
    extra={"iterations":10},
    network=Network(
        nodes=[
            Node(
                id='site-1',
                neighbors=[
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                ]
            ),
            Node(
                id='site-2',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                ]
            ),
            Node(
                id='site-3',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                ]
            ),
        ]
        )
    )
    return config
