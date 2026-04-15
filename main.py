import re
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langsmith import traceable

Max_Iterations=10
model="qwen3:1.7b"


@tool
def productPrrice(product):
    """with the product given provide the price of the product"""
    print(f"getting the price of the product {product}")
    prices={"laptop":18356,"keyboard":1232,"mouse":342}
    return prices.get(product,0)

@tool
def applyDiscount(price, discount_tier):
    """Apply discount to the price based on the discount tier.
    
    Args:
        price: The price as a number (int or float)
        discount_tier: The discount tier (gold, silver, or bronze)
    """
    print(f"Applying {discount_tier} discount")
    
    # Handle if price is passed as a dict
    if isinstance(price, dict):
        price = price.get('value', 0)
    
    # Convert to float if it's a string
    price = float(price)
    
    discounts = {"gold": 15, "silver": 10, "bronze": 5}
    discount = discounts.get(discount_tier, 0)
    discounted_price = price * (1 - discount / 100)
    return round(discounted_price, 2)

@traceable(name="langchain agent loop")
def runAgent(question):
    tools=[productPrrice,applyDiscount]
    tools_dict={t.name: t for t in tools}

    llm=init_chat_model(model, model_provider="ollama", temperature=0)
    llm_tools=llm.bind_tools(tools)

    print(f"question : {question}")
    print("="*60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to tools to help answer questions.\n\n"
                "Available tools:\n"
                "- productPrrice: Get the price of a product (laptop, keyboard, mouse)\n"
                "- applyDiscount: Apply a discount tier (gold, silver, bronze) to a price\n\n"
                "To answer the user's question:\n"
                "1. First call productPrrice to get the product price\n"
                "2. Then call applyDiscount with that price and the discount tier\n"
                "3. Finally, provide the discounted price to the user"
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, Max_Iterations + 1):
        print(f"iteration : {iteration}")
        ai_message = llm_tools.invoke(messages)
        messages.append(ai_message)
        
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        # Execute each tool call
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            print(f"Calling tool: {tool_name} with args: {tool_args}")
            
            # Get the tool function and execute it
            selected_tool = tools_dict[tool_name]
            tool_output = selected_tool.invoke(tool_args)
            
            print(f"Tool output: {tool_output}")
            
            # Add tool result to messages
            messages.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_id
                )
            )
    
    print("\nMax iterations reached without final answer")
    return "Unable to complete the task within the iteration limit"



load_dotenv()
def main():
    print("Hello from langchain agent!")
    runAgent("what is the price of a laptop after applying a silver discount")
    





if __name__ == "__main__":
    main()
