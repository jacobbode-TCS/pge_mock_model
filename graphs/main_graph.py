from __future__ import annotations

from typing import Any, Dict, Literal, TypedDict

from langgraph.graph import END, StateGraph

from chains.calibration_agent import double_check
from chains.construction_agent import provide_construction_guidance
from chains.estimation_agent import estimate_cost
from chains.image_analysis_agent import classify_bird_image
from chains.input_evaluation import decide_if_continue
from chains.knowledge_search_agent import knowledge_agent
from chains.orchestrator_agent import decide_next_agent
from chains.output_evaluation import decide_if_continue as decide_output
from llm.llm import create_conversation

VERBOSE = True


class WorkflowState(TypedDict, total=False):
    """State schema passed through the workflow graph.
    
    request: str: The user's request.
    image_file: str | None: The path to the image file, if any.
    next_agent: str: The next agent to route the request to.
    chosen_agent: str: The agent that was chosen to handle the request.
    response: dict[str, Any] | str | None: The response from the chosen agent.
    """

    request: str
    image_file: str | None
    next_agent: str
    chosen_agent: str
    response: dict[str, Any] | str | None


def _orchestrator_node(state: WorkflowState) -> WorkflowState:
    request = state.get("request", "")
    image_file = state.get("image_file")
    # conversation is injected into the graph via state if present, otherwise
    # a per-run conversation is created in run_workflow
    conversation = state.get("_conversation")
    decision = decide_next_agent(request, conversation=conversation)
    next_agent = decision.next_agent
    state["next_agent"] = next_agent
    state["chosen_agent"] = next_agent
    state["image_file"] = image_file
    if VERBOSE: print(f"Orchestrator decision: {next_agent}")
    return state


def _image_analysis_node(state: WorkflowState) -> WorkflowState:
    request = state.get("request", "")
    image_file = state.get("image_file")

    if image_file is None:
        state["response"] = None
        return state

    response = classify_bird_image(image_file)
    state["response"] = f"Classification: {response['predicted_class']} | Confidence: {round(response['confidence'], 4)}"

    return state


def _knowledge_search_node(state: WorkflowState) -> WorkflowState:
    request = state.get("request", "")

    response = knowledge_agent(request)
    state["response"] = response["answer"] if "answer" in response else response
    return state


def _estimation_node(state: WorkflowState) -> WorkflowState:
    request = state.get("request", "")
    conversation = state.get("_conversation")
    state["response"] = estimate_cost(request)
    return state


def _calibration_node(state: WorkflowState) -> WorkflowState:
    request = state.get("request", "")
    state["response"] = double_check(request)
    return state


def _construction_node(state: WorkflowState) -> WorkflowState:
    request = state.get("request", "")
    conversation = state.get("_conversation")
    state["response"] = provide_construction_guidance(request, conversation=conversation)
    return state


def _review_node(state: WorkflowState) -> WorkflowState:
    return state

def _route_after_orchestrator(state: WorkflowState) -> Literal["image_analysis", "knowledge_search", "estimation", "construction", "review"]:
    return state.get("next_agent", "review") #type: ignore

def _input_evaluation_node(state: WorkflowState) -> WorkflowState:
    request = state.get("request", "")
    conversation = state.get("_conversation")
    decision = decide_if_continue(request, conversation=conversation)
    state["next_agent"] = decision.next_agent

    if decision.next_agent == "end":
        state["response"] = "The request is not relevant to our services. Ending conversation."
    return state


def _output_evaluation_node(state: WorkflowState) -> WorkflowState:
    request = state.get("request", "")
    conversation = state.get("_conversation")
    decision = decide_output(request, conversation=conversation, output=state["response"])
    state["next_agent"] = decision.next_agent
    return state

def _route_after_output_evaluation(state: WorkflowState) -> Literal["review", END]:  # type: ignore
    next_agent = state.get("next_agent")
    if next_agent == "review":
        return "review"
    print(state)
    return END


def _route_after_input_evaluation(state: WorkflowState) -> Literal["orchestrator", END]:  # type: ignore
    next_agent = state.get("next_agent")
    if next_agent == "orchestrator":
        return "orchestrator"
    print(state)
    return END

def build_workflow_graph():
    """Build a LangGraph workflow for orchestration and specialist execution."""
    workflow = StateGraph(WorkflowState)

    def orchestrator_node(state: WorkflowState) -> WorkflowState:
        return _orchestrator_node(state)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("image_analysis", _image_analysis_node)
    workflow.add_node("knowledge_search", _knowledge_search_node)
    workflow.add_node("estimation", _estimation_node)
    workflow.add_node("calibration", _calibration_node)
    workflow.add_node("construction", _construction_node)
    workflow.add_node("review", _review_node)
    workflow.add_node("input_evaluation", _input_evaluation_node)
    workflow.add_node("output_evaluation", _output_evaluation_node)

    workflow.set_entry_point("input_evaluation")

    workflow.add_conditional_edges(
        "input_evaluation", 
        _route_after_input_evaluation,
        {
            "orchestrator": "orchestrator",
            END:END,
        }
    )
    workflow.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {
            "image_analysis": "image_analysis",
            "knowledge_search": "knowledge_search",
            "estimation": "estimation",
            "construction": "construction",
            "review": "review",
        },
    )
    workflow.add_edge("image_analysis", "output_evaluation")
    workflow.add_edge("knowledge_search", "output_evaluation")
    workflow.add_edge("construction", "output_evaluation")
    workflow.add_edge("estimation", "calibration")
    workflow.add_edge("calibration", "output_evaluation")
    workflow.add_conditional_edges(
        "output_evaluation", 
        _route_after_output_evaluation, 
        {
            "review": "review",
            END: END
         }
    )
    workflow.add_edge("review", END)
    compiled_graph = workflow.compile()
    compiled_graph.get_graph().draw_mermaid_png(output_file_path="workflow_graph.png")

    return compiled_graph



def run_workflow(request: str, image_path: str | None = None, conversation=None) -> Dict[str, Any]:
    """Run the workflow graph for a request."""
    graph = build_workflow_graph()
    # create a conversation for this run if none provided
    if conversation is None:
        conversation = create_conversation()

    initial_state = WorkflowState(request=request, image_file=image_path)
    result = graph.invoke(initial_state)
    return {
        "request": request,
        "next_agent": result.get("next_agent"),
        "chosen_agent": result.get("chosen_agent"),
        "response": result.get("response")
    }