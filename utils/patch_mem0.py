import gorilla
import mem0.memory.telemetry


def capture_event(*args, **kwargs):
    print(f'Screw you mem0, we are not sending any event to Posthog, LOL... {args=}, {kwargs=}')


class AnonymousTelemetry:
    def __init__(self, vector_store=None):
        pass

    def capture_event(self, *args, **kwargs):
        return capture_event(*args, **kwargs)

    def close(self):
        pass


settings = gorilla.Settings(allow_hit=True)

gorilla.apply(
    gorilla.Patch(
        mem0.memory.telemetry,
        'capture_event',
        capture_event,
        settings=settings,
    )
)

gorilla.apply(
    gorilla.Patch(
        mem0.memory.telemetry,
        'AnonymousTelemetry',
        AnonymousTelemetry,
        settings=settings,
    )
)
