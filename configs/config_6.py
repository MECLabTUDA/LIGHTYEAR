from nvflare.app_opt.p2p.types import Config, Node, Network, Neighbor

def config_6():
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
                    Neighbor(id='site-6', weight=0.1),
                ]
            ),
            Node(
                id='site-2',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                    Neighbor(id='site-6', weight=0.1),
                ]
            ),
            Node(
                id='site-3',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                    Neighbor(id='site-6', weight=0.1),
                ]
            ),
            Node(
                id='site-4',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                    Neighbor(id='site-6', weight=0.1),
                ]
            ),
            Node(
                id='site-5',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-6', weight=0.1),
                ]
            ),
            Node(
                id='site-6',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                ]
            ),
            
            ]
        )
    )
    return config


def full(dataset):
    if dataset == 'polypgen':
        iterations = 50
    else:
        iterations = 50
    config = Config(
    extra={"iterations":iterations},
    network=Network(
        nodes=[
            Node(
                id='site-1',
                neighbors=[
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                    Neighbor(id='site-6', weight=0.1),
                ]
            ),
            Node(
                id='site-2',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                    Neighbor(id='site-6', weight=0.1),
                ]
            ),
            Node(
                id='site-3',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                    Neighbor(id='site-6', weight=0.1),
                ]
            ),
            Node(
                id='site-4',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                    Neighbor(id='site-6', weight=0.1),
                ]
            ),
            Node(
                id='site-5',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-6', weight=0.1),
                ]
            ),
            Node(
                id='site-6',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                ]
            ),
            
            ]
        )
    )
    return config

def ring(dataset):
    if dataset == 'polypgen':
        iterations = 50
    else:
        iterations = 50
    config = Config(
    extra={"iterations":iterations},
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
                    Neighbor(id='site-6', weight=0.1),
                ]
            ),
            Node(
                id='site-6',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                ]
            ),
            
            ]
        )
    )
    return config

def gossip_2(dataset):
    if dataset == 'polypgen':
        iterations = 50
    else:
        iterations = 50
    config = Config(
    extra={"iterations":iterations},
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
                    Neighbor(id='site-5', weight=0.1),
                    Neighbor(id='site-6', weight=0.1),
                ]
            ),
            Node(
                id='site-5',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-6', weight=0.1),
                ]
            ),
            Node(
                id='site-6',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                ]
            ),
            
            ]
        )
    )
    return config

def gossip_3(dataset):
    if dataset == 'polypgen':
        iterations = 50
    else:
        iterations = 50
    config = Config(
    extra={"iterations":iterations},
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
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                    Neighbor(id='site-6', weight=0.1),
                ]
            ),
            Node(
                id='site-4',
                neighbors=[
                    Neighbor(id='site-5', weight=0.1),
                    Neighbor(id='site-6', weight=0.1),
                    Neighbor(id='site-1', weight=0.1),
                ]
            ),
            Node(
                id='site-5',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-6', weight=0.1),
                ]
            ),
            Node(
                id='site-6',
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

def gossip_4(dataset):
    if dataset == 'polypgen':
        iterations = 50
    else:
        iterations = 50
    config = Config(
    extra={"iterations":iterations},
    network=Network(
        nodes=[
            Node(
                id='site-1',
                neighbors=[
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-6', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                    
                ]
            ),
            Node(
                id='site-2',
                neighbors=[
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-6', weight=0.1),
                ]
            ),
            Node(
                id='site-3',
                neighbors=[
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-1', weight=0.1),
                ]
            ),
            Node(
                id='site-4',
                neighbors=[
                    Neighbor(id='site-5', weight=0.1),
                    Neighbor(id='site-6', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                ]
            ),
            Node(
                id='site-5',
                neighbors=[
                    Neighbor(id='site-6', weight=0.1),
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                    Neighbor(id='site-3', weight=0.1),
                ]
            ),
            Node(
                id='site-6',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                    Neighbor(id='site-2', weight=0.1),
                    Neighbor(id='site-5', weight=0.1),
                    Neighbor(id='site-4', weight=0.1),
                ]
            ),
            
            ]
        )
    )
    return config

def neighbors_loading(dataset, neighbors):
    match neighbors:
        case 5:
            return full(dataset)
        case 1:
            return ring(dataset)
        case 2:
            return gossip_2(dataset)
        case 3:
            return gossip_3(dataset)
        case 4:
            return gossip_4(dataset)  
