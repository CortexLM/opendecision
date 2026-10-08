"""HTTP surface (extra: opendecision[serve]). Laya-compatible paths."""
from .decider import Decider


def create_app(decider: Decider | None = None):
    from fastapi import FastAPI, HTTPException
    from pydantic import ValidationError
    d = decider or Decider()
    app = FastAPI(title="OpenDecision")

    @app.get("/health")
    def health():
        return {"ok": True, "model": d.version}

    @app.post("/v1/systemone")
    def systemone(body: dict):
        try:
            return d.predict(body)
        except ValidationError as e:
            raise HTTPException(422, e.errors(include_url=False, include_context=False))

    @app.post("/v1/systemone/batch")
    def batch(body: list[dict]):
        return [d.predict(b) for b in body]
    return app
