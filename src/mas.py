import datetime
import os, json, re
from typing import Any, Coroutine
from jinja2 import Environment, FileSystemLoader
from openai import AzureOpenAI
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import (
    KernelFunctionSelectionStrategy,
)
from semantic_kernel.agents.strategies.termination.kernel_function_termination_strategy import (
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from semantic_kernel.connectors.search_engine import BingConnector
from semantic_kernel.core_plugins import WebSearchEnginePlugin
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.filters.filter_types import FilterTypes
from collections.abc import Coroutine
from typing import Any
from semantic_kernel.filters.functions.function_invocation_context import FunctionInvocationContext
from semantic_kernel.core_plugins.sessions_python_tool.sessions_python_plugin import SessionsPythonTool

from azure.core.credentials import AccessToken
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential
from semantic_kernel.exceptions.function_exceptions import FunctionExecutionException

from src.plugins.presentation import PresentationPlugin

class Orchestrator:
    def __init__(self, user_input):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-12-01-preview"
        )
        
        self.env = Environment(loader=FileSystemLoader(os.getenv('TEMPLATE_DIR_PROMPTS')))
        self.template = self.env.get_template(os.getenv('TEMPLATE_SYSTEM_ORCHESTRATOR'))
        self.theme = user_input

    def get_response(self):
        response = self.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL_ORCHESTRATOR"),
            messages=[
                {"role": "user", "content": self.template.render(theme=self.theme)},
            ],
            max_completion_tokens=5000
        )
        return response

    def parse_response(self, response):
        json_response = json.loads(response.model_dump_json())
        json_response = json.loads(json_response['choices'][0]['message']['content'].replace('```json\n', '').replace('```', ''))
        return json_response

    def get_dynamic_agents(self, json_response):
        agents = []
        for agent_info in json_response.get('agents', []):
            agent = {
                'name': agent_info['name'],
                'role': agent_info['role'],
                'system_prompt': agent_info['system_prompt'],
            }
            agents.append(agent)
        return agents

    def run(self):
        response = self.get_response()
        json_response = self.parse_response(response)
        print(f'Creating dynamic agents who will be responsible for creating the presentation about: {self.theme}')
        dynamic_agents = self.get_dynamic_agents(json_response)
        return dynamic_agents

class MultiAgent:
    def __init__(self):
        self.project_client = AIProjectClient.from_connection_string(credential=DefaultAzureCredential(),
                                                                     conn_str=os.environ["PROJECT_CONNECTION_STRING"])
        self.model = os.getenv("AZURE_OPENAI_MODEL")
        self.bing_connector = BingConnector(os.getenv("BING_API_KEY"))
    
    @staticmethod
    def _create_kernel_with_chat_completion(service_id: str) -> Kernel:
        kernel = Kernel()
        kernel.add_service(AzureChatCompletion(service_id=service_id))
        return kernel
    
    @staticmethod
    def _standardize_string(input_string: str) -> str:
        return re.sub(r'[^0-9A-Za-z_-]', '_', input_string)
    
    def create_agents(self, dynamic_agents):
        expert_agents = []
        for agent in dynamic_agents:

            agent_name = self._standardize_string(agent['name'])
            kernel = self._create_kernel_with_chat_completion(agent_name)            
        
            # Configure the function choice behavior to auto invoke kernel functions
            settings = kernel.get_prompt_execution_settings_from_service_id(service_id=agent_name)
            settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

            kernel.add_plugin(WebSearchEnginePlugin(BingConnector()), "WebSearch")
            kernel.add_plugin(PresentationPlugin(), "Presentation")
           
            expert = ChatCompletionAgent(service_id=agent_name,
                                         kernel=kernel,
                                         name=agent_name,
                                         instructions=agent['system_prompt'],
                                         execution_settings=settings            
                                        )
            expert_agents.append(expert)

            print(f"Created agent: {expert.name}, agent ID: {expert.id} ")
        return expert_agents
    
    def create_selection_function(self, expert_agents):
        selection_function = KernelFunctionFromPrompt(function_name="selection",
                                                        prompt=f"""
                                                        Determine which participant takes the next turn in a conversation based on the the most recent participant.
                                                        State only the name of the participant to take the next turn.
                                                        No participant should take more than one turn in a row.

                                                        Choose only from these participants:
                                                        {expert_agents}

                                                        History:
                                                        {{{{$history}}}}
                                                        """)
        return selection_function
    
    def create_termination_function(self, termination_keyword):
        selection_function = KernelFunctionFromPrompt(function_name="termination",
                                                      prompt= f""" 
                                                        Examine the RESPONSE and determine whether the content has been deemed satisfactory.
                                                        If content is satisfactory, respond with a single word without explanation: {termination_keyword}.
                                                        If specific suggestions are being provided, it is not satisfactory.
                                                        If no correction is suggested, it is satisfactory.

                                                        RESPONSE:
                                                        {{{{$history}}}}
              """)
        return selection_function

    def create_chat_group(self, expert_agents, selection_function, termination_function, termination_keyword):
        group = AgentGroupChat(agents=expert_agents,
                               selection_strategy=KernelFunctionSelectionStrategy(
                                    function=selection_function,
                                    kernel=self._create_kernel_with_chat_completion("selection"),
                                    result_parser=lambda result: str(result.value[0]) if result.value is not None else expert_agents[-1].name,
                                    agent_variable_name="agents",
                                    history_variable_name="history",
                                ),
                                termination_strategy=KernelFunctionTerminationStrategy(
                                    agents=[expert_agents[-1]],
                                    function=termination_function,
                                    kernel=self._create_kernel_with_chat_completion("termination"),
                                    result_parser=lambda result: termination_keyword in str(result.value[0]).lower(),
                                    history_variable_name="history",
                                    maximum_iterations=10,
                                ),
                        )
          
        return group
    
    

    def auth_callback_factory(self, scope):
        auth_token = None
        async def auth_callback() -> str:
            """Auth callback for the SessionsPythonTool.
            This is a sample auth callback that shows how to use Azure's DefaultAzureCredential
            to get an access token.
            """
            nonlocal auth_token
            current_utc_timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())

            if not auth_token or auth_token.expires_on < current_utc_timestamp:
                credential = DefaultAzureCredential()

                try:
                    auth_token = credential.get_token(scope)
                except ClientAuthenticationError as cae:
                    err_messages = getattr(cae, "messages", [])
                    raise FunctionExecutionException(
                        f"Failed to retrieve the client auth token with messages: {' '.join(err_messages)}"
                    ) from cae

            return auth_token.token
        
        return auth_callback
