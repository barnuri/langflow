from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from langflow.services.flow.flow_runner import LangflowRunnerExperimental


@pytest.fixture
def sample_flow_dict():
    return {
        "id": str(uuid4()),  # Add required ID field
        "name": "test_flow",  # Add name field
        "data": {
            "nodes": [],
            "edges": [],
        },
    }


@pytest.fixture
def flow_runner():
    return LangflowRunnerExperimental()


@pytest.mark.asyncio
async def test_database_exists_check(flow_runner):
    """Test database exists check functionality."""
    result = await flow_runner.database_exists_check()
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_get_flow_dict_from_dict(flow_runner, sample_flow_dict):
    """Test loading flow from a dictionary."""
    result = await flow_runner.get_flow_dict(sample_flow_dict)
    assert result == sample_flow_dict


@pytest.mark.asyncio
async def test_get_flow_dict_invalid_input(flow_runner):
    """Test loading flow with invalid input type."""
    pattern = r"Input must be a file path .* or a JSON object .*"
    with pytest.raises(TypeError, match=pattern):
        await flow_runner.get_flow_dict(123)


@pytest.mark.asyncio
async def test_run_with_dict_input(flow_runner, sample_flow_dict):
    """Test running flow with dictionary input."""
    session_id = str(uuid4())
    input_value = "test input"

    result = await flow_runner.run(
        session_id=session_id,
        flow=sample_flow_dict,
        input_value=input_value,
    )
    assert result is not None


@pytest.mark.asyncio
async def test_run_with_different_input_types(flow_runner, sample_flow_dict):
    """Test running flow with different input and output types."""
    session_id = str(uuid4())
    test_cases = [
        ("text input", "text", "text"),
        ("chat input", "chat", "chat"),
        ("test input", "chat", "all"),  # Updated to use "all" as default output_type
    ]

    for input_value, input_type, output_type in test_cases:
        result = await flow_runner.run(
            session_id=session_id,
            flow=sample_flow_dict,
            input_value=input_value,
            input_type=input_type,
            output_type=output_type,
        )
        assert result is not None


@pytest.mark.asyncio
async def test_initialize_database(flow_runner):
    """Test database initialization."""
    flow_runner.should_initialize_db = True
    await flow_runner.init_db_if_needed()
    assert not flow_runner.should_initialize_db


def test_process_tweaks_no_vertices(flow_runner):
    """Test process_tweaks with no vertices."""
    flow_dict = {"id": str(uuid4()), "name": "test_flow", "data": {}}
    
    with patch("langflow.services.flow.flow_runner.Graph.from_payload") as mock_graph:
        mock_graph.return_value.vertices = []
        result = flow_runner.process_tweaks(flow_dict)
        assert result == flow_dict


def test_process_tweaks_with_vertices_no_load_from_db(flow_runner):
    """Test process_tweaks with vertices that don't need loading from DB."""
    flow_dict = {"id": str(uuid4()), "name": "test_flow", "data": {}}
    
    mock_vertex = MagicMock()
    mock_vertex.id = "vertex1"
    
    with patch("langflow.services.flow.flow_runner.Graph.from_payload") as mock_graph, \
         patch("langflow.services.flow.flow_runner.ParameterHandler") as mock_param_handler:
        
        mock_graph.return_value.vertices = [mock_vertex]
        mock_handler_instance = mock_param_handler.return_value
        mock_handler_instance.process_field_parameters.return_value = ({}, [])
        
        result = flow_runner.process_tweaks(flow_dict)
        assert result == flow_dict
        mock_param_handler.assert_called_once_with(mock_vertex, None)


def test_process_tweaks_with_load_from_db_fields(flow_runner):
    """Test process_tweaks with vertices that have load_from_db fields."""
    flow_dict = {"id": str(uuid4()), "name": "test_flow", "data": {}}
    
    mock_vertex = MagicMock()
    mock_vertex.id = "vertex1"
    
    with patch("langflow.services.flow.flow_runner.Graph.from_payload") as mock_graph, \
         patch("langflow.services.flow.flow_runner.ParameterHandler") as mock_param_handler, \
         patch("langflow.services.flow.flow_runner.replace_tweaks_with_env") as mock_replace, \
         patch("langflow.services.flow.flow_runner.process_tweaks") as mock_process:
        
        mock_graph.return_value.vertices = [mock_vertex]
        mock_handler_instance = mock_param_handler.return_value
        field_params = {"field1": "value1", "field2": ""}
        load_from_db_fields = ["field1", "field2"]
        mock_handler_instance.process_field_parameters.return_value = (field_params, load_from_db_fields)
        
        mock_replace.return_value = {"vertex1": {"field1": "env_value1"}}
        mock_process.return_value = {"processed": True}
        
        result = flow_runner.process_tweaks(flow_dict)
        
        assert result == {"processed": True}
        mock_param_handler.assert_called_once_with(mock_vertex, None)
        mock_replace.assert_called_once()
        mock_process.assert_called_once()


def test_process_tweaks_load_from_db_update(flow_runner):
    """Test that process_tweaks correctly updates load_from_db fields to False."""
    flow_dict = {
        "id": str(uuid4()),
        "name": "test_flow",
        "data": {
            "nodes": [
                {
                    "template": {
                        "field1": {"load_from_db": True, "value": "test"},
                        "field2": {"load_from_db": False, "value": "test2"}
                    }
                }
            ]
        }
    }
    
    with patch("langflow.services.flow.flow_runner.Graph.from_payload") as mock_graph, \
         patch("langflow.services.flow.flow_runner.ParameterHandler") as mock_param_handler:
        
        mock_graph.return_value.vertices = []
        mock_param_handler.return_value.process_field_parameters.return_value = ({}, [])
        
        result = flow_runner.process_tweaks(flow_dict)
        
        # Check that load_from_db was updated to False
        node_template = result["data"]["nodes"][0]["template"]
        assert node_template["field1"]["load_from_db"] is False
        assert node_template["field2"]["load_from_db"] is False


def test_process_tweaks_nested_load_from_db_update(flow_runner):
    """Test that process_tweaks recursively updates nested load_from_db fields."""
    flow_dict = {
        "id": str(uuid4()),
        "name": "test_flow", 
        "data": {
            "nested": {
                "deep": {
                    "load_from_db": True,
                    "other_field": "value"
                },
                "list": [
                    {"load_from_db": True, "value": "test1"},
                    {"load_from_db": False, "value": "test2"}
                ]
            }
        }
    }
    
    with patch("langflow.services.flow.flow_runner.Graph.from_payload") as mock_graph, \
         patch("langflow.services.flow.flow_runner.ParameterHandler") as mock_param_handler:
        
        mock_graph.return_value.vertices = []
        mock_param_handler.return_value.process_field_parameters.return_value = ({}, [])
        
        result = flow_runner.process_tweaks(flow_dict)
        
        # Check that nested load_from_db fields were updated
        assert result["data"]["nested"]["deep"]["load_from_db"] is False
        assert result["data"]["nested"]["list"][0]["load_from_db"] is False 
        assert result["data"]["nested"]["list"][1]["load_from_db"] is False


def test_process_tweaks_multiple_vertices_with_tweaks(flow_runner):
    """Test process_tweaks with multiple vertices having different tweak scenarios."""
    flow_dict = {"id": str(uuid4()), "name": "test_flow", "data": {}}
    
    mock_vertex1 = MagicMock()
    mock_vertex1.id = "vertex1"
    mock_vertex2 = MagicMock() 
    mock_vertex2.id = "vertex2"
    
    with patch("langflow.services.flow.flow_runner.Graph.from_payload") as mock_graph, \
         patch("langflow.services.flow.flow_runner.ParameterHandler") as mock_param_handler, \
         patch("langflow.services.flow.flow_runner.replace_tweaks_with_env") as mock_replace, \
         patch("langflow.services.flow.flow_runner.process_tweaks") as mock_process:
        
        mock_graph.return_value.vertices = [mock_vertex1, mock_vertex2]
        
        # Mock different responses for each vertex
        def param_handler_side_effect(vertex, storage):
            mock_instance = MagicMock()
            if vertex.id == "vertex1":
                mock_instance.process_field_parameters.return_value = ({"field1": "value1"}, ["field1"])
            else:  # vertex2
                mock_instance.process_field_parameters.return_value = ({"field2": "value2"}, ["field2"])
            return mock_instance
        
        mock_param_handler.side_effect = param_handler_side_effect
        mock_replace.return_value = {"vertex1": {"field1": "value1"}, "vertex2": {"field2": "value2"}}
        mock_process.return_value = {"processed": True}
        
        result = flow_runner.process_tweaks(flow_dict)
        
        assert result == {"processed": True}
        assert mock_param_handler.call_count == 2
        mock_replace.assert_called_once()
        mock_process.assert_called_once()
