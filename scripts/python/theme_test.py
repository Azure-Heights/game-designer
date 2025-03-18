
# +-- imports

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser

# --+


theme_prompt = ChatPromptTemplate.from_messages([# +--
    ("system",
     """You are a creative board game designer. Your task is to generate a compelling and unique theme for a board game.

     **Guidelines:**
     - The theme should be engaging and imaginative.
     - It can be based on real-world history, science fiction, fantasy, or abstract concepts.
     - Avoid overly generic themes like "medieval knights" unless they have a twist.

     ### Example Outputs:
     - "Time-traveling archaeologists recovering lost artifacts."
     - "A dystopian city where players are rebel hackers fighting against an AI overlord."
     - "A fantasy world where players control rival dragon clans competing for magical dominance."
     """),
    ("human", "Generate a unique board game theme."),
])# --+


genre_prompt = ChatPromptTemplate.from_messages([# +--
    ("system",
     """ You are designing a board game with the theme: **{theme}**.

     Now, decide the most fitting **game genre** that complements this theme.

     **Guidelines:**
     - Choose from genres like Strategy, Party, Cooperative, Deduction, Economic, Exploration, etc.
     - Consider how the theme influences gameplay.
     - The genre should shape the player experience.

     ### Example Outputs:
     - Theme: "Time-traveling archaeologists" → Genre: "Exploration, Strategy"
     - Theme: "Dystopian hackers fighting an AI overlord" → Genre: "Cooperative, Social Deduction"
     - Theme: "Rival dragon clans in a fantasy world" → Genre: "Area Control, Combat"
     """),
    ("human", "What is the best genre for the theme \"{theme}\"?"),
])# --+


mechanics_prompt = ChatPromptTemplate.from_messages([# +--
    ("system",
     """ You are designing a board game with the theme **{theme}** and the genre **{genre}**.

     Now, determine the **best core mechanics** that suit the game.

     **Guidelines:**
     - Choose 1-3 primary mechanics (e.g., Deck-building, Worker Placement, Hidden Roles, Dice Rolling, Area Control).
     - Make sure the mechanics reinforce the theme and genre.
     - If possible, suggest unique twists on common mechanics.

     ### Example Outputs:
     - Theme: "Time-traveling archaeologists" → Mechanics: "Tile Placement, Set Collection, Push Your Luck"
     - Theme: "Dystopian hackers fighting an AI overlord" → Mechanics: "Hidden Roles, Bluffing, Resource Management"
     - Theme: "Rival dragon clans in a fantasy world" → Mechanics: "Area Control, Asymmetric Abilities"

     """),
    ("human", "What are the best mechanics for **{theme}** with the genre **{genre}**?"),
])# --+


llm = ChatOpenAI(model="gpt-4o")
chain = (
    { "theme": (theme_prompt | llm | StrOutputParser()) }
    | RunnablePassthrough.assign(genre=genre_prompt | llm | StrOutputParser())
    | RunnablePassthrough.assign(mechanics=mechanics_prompt | llm | StrOutputParser())
)


if __name__ == "__main__":

    print(chain.invoke({ }))

