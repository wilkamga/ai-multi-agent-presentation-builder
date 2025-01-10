import asyncio
import streamlit as st
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from src.mas import Orchestrator, MultiAgent

# Apply a custom page config for a better visual experience
st.set_page_config(
    page_title="AI Multi-Agent Presentation Builder",
    page_icon=":robot_face:",
    layout="wide"
)

# Add some basic style
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #e0eafc, #cfdef3);
        color: #333;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 1em;
        margin-top: 1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

async def run(user_input):
    orchestrator = Orchestrator(user_input)
    dynamic_agents = orchestrator.run()

    mas = MultiAgent()
    expert_agents = mas.create_agents(dynamic_agents)
    expert_agents_names = [agent.name for agent in expert_agents]

    with st.sidebar:
        st.subheader("Expert Agents")
        st.markdown("Creating the Expert Agents")

        for agent_name in expert_agents_names:
            st.info(agent_name)
            # Add a 0.5 second delay
            await asyncio.sleep(0.5)

    selection_function = mas.create_selection_function(expert_agents_names)
    termination_keyword = 'yes'
    termination_function = mas.create_termination_function(termination_keyword)

    group = mas.create_chat_group(
        expert_agents,
        selection_function,
        termination_function,
        termination_keyword
    )
    return group

async def main(user_input):
    group = await run(user_input)
    await group.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))

    async for response in group.invoke():
        st.markdown(f"**{response.role} - {response.name or '*'}**")
        st.info(response.content)

        if group.is_complete:
            st.success("Conversation completed!")
            break

def main_app():
    st.title("AI Multi-Agent Presentation Builder")
    st.subheader("Create a stunning presentation with AI agents")
    user_input = st.text_input("Enter the theme for the presentation:")

    if st.button("Create Presentation"):
        if user_input:
            asyncio.run(main(user_input))
        else:
            st.warning("Please provide a theme before creating a presentation.")

if __name__ == "__main__":
    main_app()