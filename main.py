from itertools import chain
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama  import OllamaLLM

import os

load_dotenv()

def main():
    print("Hello from langchain!")
    #print(os.environ.get("HUGGINGFACEHUB_API_TOKEN"))

    information = """
    India accounts for the bulk of the Indian subcontinent, lying atop the Indian tectonic plate, and a part of the Indo-Australian Plate.[161] India's defining geologic processes began approximately 70 million years ago, when the Indian Plate, then part of the southern supercontinent Gondwana, began a north-eastward drift caused by seafloor spreading to its south-west, and later, south and south-east.[161] Simultaneously, the vast Tethyan oceanic crust, to its northeast, began to subduct under the Eurasian Plate.[161] The Indian continental crust, however, was obstructed and was sheared horizontally. Its lower crust and mantle slid under, but the upper layer piled up in sheets (or nappes) ahead of the subduction zone.[162] This created the orogeny, or process of mountain building, of the Himalayas.[163] The middle and stiffer layer continued to push into Tibet, causing crustal thickening of the Tibetan Plateau.[164] Immediately south of the emerging Himalayas, plate movement created a vast crescent-shaped trough that rapidly filled with river-borne sediment[165] and now constitutes the Indo-Gangetic Plain.[166] The original Indian plate makes its first appearance above the sediment in the ancient Aravalli range, which extends from the Delhi Ridge in a southwesterly direction. To the west lies the Thar Desert, the eastern spread of which is checked by the Aravallis.[167][168][169]
    """
    summaryTemplate = f"""
    From the {information} given about indian Geography provide the below:
    1.summary of the information
    2.important points
    """
    summaryPromptTemplate = PromptTemplate(input_variables=["information"], template=summaryTemplate)

    llm = OllamaLLM(model ="llama2",temperature= 0.3)

    chain =summaryPromptTemplate | llm
    response =chain.invoke({"information": information})
    print(response)

if __name__ == "__main__":
    main()
