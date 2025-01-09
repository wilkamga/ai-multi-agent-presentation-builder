import asyncio, os
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from src.mas import MultiAgent, Orchestrator


async def run(user_input):
    # Create the orchestrator
    orchestrator = Orchestrator(user_input)
    dynamic_agents = orchestrator.run()

    mas = MultiAgent()

    # Create agents
    expert_agents = mas.create_agents(dynamic_agents)
    expert_agents_names = [agent.name for agent in expert_agents]

    # Create selection function
    selection_function = mas.create_selection_function(expert_agents_names)

    # Create the termination function
    termination_keyword = 'yes'
    termination_function = mas.create_termination_function(termination_keyword)

    # Create the chat group
    group = mas.create_chat_group(expert_agents, 
                                  selection_function, 
                                  termination_function, 
                                  termination_keyword)
    
    return group

async def main():   
    
    is_complete: bool = False
    while not is_complete:
        user_input = input("User:> ")
        if not user_input:
            continue

        if user_input.lower() == "exit":
            is_complete = True
            break
        
        # Run the orchestrator and get the chat group
        group = await run(user_input)
    
        if user_input.lower() == "reset":
            await group.reset()
            print("[Conversation has been reset]")
            continue

        if user_input.startswith("@") and len(input) > 1:
            file_path = input[1:]
            try:
                if not os.path.exists(file_path):
                    print(f"Unable to access file: {file_path}")
                    continue
                with open(file_path) as file:
                    user_input = file.read()
            except Exception:
                print(f"Unable to access file: {file_path}")
                continue

        await group.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))

        async for response in group.invoke():
            print(f"'***** {response.role} - {response.name or '*'} ***** '" )
            print('*****************************')
            print('\n')
            print('\n')
            
            print(f"'{response.content}'")
            print('----------------------------------')
            print('\n')
            print('\n')

        if group.is_complete:
            is_complete = True
            break

if __name__ == "__main__":
    asyncio.run(main())