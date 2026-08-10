from nvflare.app_opt.p2p.types import Config, Node, Network, Neighbor

def config_seg():
    config = Config(
    extra={"iterations":60},
    network=Network(
        nodes=[
            Node(
                id='site-1',
                neighbors=[
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                ]
            ),
            Node(
                id='site-2',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                ]
            ),
            Node(
                id='site-3',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                ]
            ),
            Node(
                id='site-4',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                ]
            ),
            Node(
                id='site-5',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                ]
                ),
            ]
        )
    )
    return config
