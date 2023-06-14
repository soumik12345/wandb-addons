from typing import Optional

import wandb
from wandb.sdk.data_types.trace_tree import (
    SpanKind,
    StatusCode,
    Result,
    Span,
    WBTraceTree,
)


class Trace:
    """Manage and log a trace - a collection of spans their metadata and hierarchy.

    Arguments:
        name: (str) The name of the root span.
        kind: (str, optional) The kind of the root span.
        status_code: (str, optional) The status of the root span, either "error" or "success".
        status_message: (str, optional) Any status message associated with the root span.
        metadata: (dict, optional) Any additional metadata for the root span.
        start_time_ms: (int, optional) The start time of the root span in milliseconds.
        end_time_ms: (int, optional) The end time of the root span in milliseconds.
        inputs: (dict, optional) The named inputs of the root span.
        outputs: (dict, optional) The named outputs of the root span.
        model_dict: (dict, optional) A json serializable dictionary containing the model architecture details.

    """

    def __init__(
        self,
        name: str,
        kind: str = None,
        status_code: str = None,
        status_message: str = None,
        metadata: dict = None,
        start_time_ms: int = None,
        end_time_ms: int = None,
        inputs: dict = None,
        outputs: dict = None,
        model_dict: dict = None,
    ):

        self._span = self._assert_and_create_span(
            name=name,
            kind=kind,
            status_code=status_code,
            status_message=status_message,
            metadata=metadata,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            inputs=inputs,
            outputs=outputs,
        )
        if model_dict is not None:
            assert isinstance(model_dict, dict), "Model dict must be a dictionary"
        self._model_dict = model_dict

    def _assert_and_create_span(
        self,
        name: str,
        kind: Optional[str] = None,
        status_code: Optional[str] = None,
        status_message: Optional[str] = None,
        metadata: Optional[dict] = None,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        inputs: Optional[dict] = None,
        outputs: Optional[dict] = None,
    ):
        if kind is not None:
            assert (
                kind.upper() in SpanKind.__members__
            ), "Invalid span kind, can be one of 'LLM', 'AGENT', 'CHAIN', 'TOOL'"
            kind = SpanKind(kind.upper())
        if status_code is not None:
            assert (
                status_code.upper() in StatusCode.__members__
            ), "Invalid status code, can be one of 'SUCCESS' or 'ERROR'"
            status_code = StatusCode(status_code.upper())
        if inputs is not None and outputs is not None:
            assert isinstance(inputs, dict), "Inputs must be a dictionary"
            assert isinstance(outputs, dict), "Outputs must be a dictionary"
            result = Result(inputs=inputs, outputs=outputs)
        else:
            result = None

        return Span(
            name=name,
            span_kind=kind,
            status_code=status_code,
            status_message=status_message,
            attributes=metadata,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            results=[result],
        )

    def add_child(
        self,
        child: "Trace",
    ) -> "Trace":
        """Add a child span to the current span of the trace."""
        self._span.add_child_span(child._span)
        if self._model_dict is not None and child._model_dict is not None:
            self._model_dict.update({child._span.name: child._model_dict})
        return self

    def add_metadata(self, metadata: dict) -> "Trace":
        """Add metadata to the span of the current trace."""
        if self._span.attributes is None:
            self._span.attributes = metadata
        else:
            self._span.attributes.update(metadata)
        return self

    def add_inputs_and_outputs(self, inputs: dict, outputs: dict) -> "Trace":
        """Add a result to the span of the current trace."""
        if self._span.results == [None]:
            result = Result(inputs=inputs, outputs=outputs)
            self._span.results = [result]
        else:
            result = Result(inputs=inputs, outputs=outputs)
            self._span.results.append(result)
        return self

    def log(self, name: str) -> None:
        """Log the trace to a wandb run"""
        trace_tree = WBTraceTree(self._span, self._model_dict)
        assert (
            wandb.run is not None
        ), "You must call wandb.init() before logging a trace"
        wandb.run.log({name: trace_tree})
