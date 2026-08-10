from nvflare.app_opt.p2p.types import Config, Node, Network, Neighbor

def config_5():
    config = Config(
    extra={"iterations":12},
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


def ring():
    config = Config(
    extra={"iterations":20},
    network=Network(
        nodes=[
            Node(
                id='site-1',
                neighbors=[
                    Neighbor(id='site-2', weight=0.1),
                ]
            ),
            Node(
                id='site-2',
                neighbors=[
                    Neighbor(id='site-3', weight=0.1),
                ]
            ),
            Node(
                id='site-3',
                neighbors=[
                    Neighbor(id='site-4', weight=0.1),
                ]
            ),
            Node(
                id='site-4',
                neighbors=[
                    Neighbor(id='site-5', weight=0.1),
                ]
            ),
            Node(
                id='site-5',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                ]
                ),
            ]
        )
    )
    return config

def gossip_2():
    config = Config(
    extra={"iterations":20},
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
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                ]
            ),
            Node(
                id='site-3',
                neighbors=[
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                ]
            ),
            Node(
                id='site-4',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                ]
            ),
            Node(
                id='site-5',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                ]
                ),
            ]
        )
    )
    return config

def gossip_3():
    config = Config(
    extra={"iterations":20},
    network=Network(
        nodes=[
            Node(
                id='site-1',
                neighbors=[
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                ]
            ),
            Node(
                id='site-2',
                neighbors=[
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                ]
            ),
            Node(
                id='site-3',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                ]
            ),
            Node(
                id='site-4',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                ]
            ),
            Node(
                id='site-5',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                ]
                ),
            ]
        )
    )
    return config

def full():
    config = Config(
    extra={"iterations":20},
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

def neighbors_loading(neighbors):
    match neighbors:
        case 4:
            return full()
        case 1:
            return ring()
        case 2:
            return gossip_2()
        case 3:
            return gossip_3()