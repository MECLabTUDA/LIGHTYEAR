from nvflare.apis.controller_spec import Task
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_opt.p2p.types import Config
from nvflare.app_opt.p2p.controllers import DistOptController

class Camelyon17Controller(DistOptController):
    def __init__(
        self,
        config: Config,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)
        self.config = config

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        # Send network config (aka neighors info) to each client
        for node in self.config.network.nodes:
            task = Task(
                name="config",
                data=DXO(
                    data_kind=DataKind.APP_DEFINED,
                    data={"neighbors": [n.__dict__ for n in node.neighbors]},
                ).to_shareable(),
            )
            self.log_info(fl_ctx, "Send config to client")
            self.send_and_wait(task=task, targets=[node.id], fl_ctx=fl_ctx)

        self.log_info(fl_ctx, "After config distribution")

        # Run algorithm (with extra params if any passed as data)
        targets = [node.id for node in self.config.network.nodes]
        self.broadcast_and_wait(
            task=Task(
                name="run_algorithm",
                data=DXO(
                    data_kind=DataKind.APP_DEFINED,
                    data={key: value for key, value in self.config.extra.items()},
                ).to_shareable(),
            ),
            targets=targets,
            min_responses=0,
            fl_ctx=fl_ctx,
        )
