import numpy as np  # type: ignore
import faiss  # type: ignore
import wikipediaapi  # type: ignore
import torch

from src import Roberta, Config, SourceMapping, Embeddings, Index, Wiki
from transformers import RobertaTokenizer, RobertaForMaskedLM

tokenizer, model = Roberta.get_default()



def get_page_section_from_wiki(source: str) -> str:
    wikipedia = wikipediaapi.Wikipedia("en")

    title = source.split("https://en.wikipedia.org/wiki/")[1].split("#")[0].replace("_"," ")

    target_page = wikipedia.page(title)
    section = target_page.section_by_title\
        (source.split("https://en.wikipedia.org/wiki/")[1].split("#")[1].replace("_"," ")) # возврощает последнюю секцию

    return str(section)


def get_prediction(sent, token_ids, masked_pos, token): # first token
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    modelMLM = RobertaForMaskedLM.from_pretrained('roberta-large')

    with torch.no_grad():
        output = modelMLM(token_ids)

    last_hidden_state = output[0].squeeze()

    list_of_list = []
    for index, mask_index in enumerate(masked_pos):
        probs = torch.nn.functional.softmax(last_hidden_state[mask_index])
        mask_hidden_state = last_hidden_state[mask_index]
        idx = torch.topk(mask_hidden_state, k=10, dim=0)[1]
        words = [tokenizer.decode(i.item()).strip() for i in idx]
        word_probs = [probs[i] for i in idx]
        list_of_list.append(words)

        indices = [i for i, x in enumerate(words) if x == token]
        for i, indec in enumerate(indices):
            print("probability:",word_probs[indec].item())
            print("Mask position", mask_index, "Guesses : ", words[indec],"\n")
        # token_with_c = "'"+token+"'"
        # if token in words:
        #     print("probability:",word_probs.index(token))
        #     print("Mask position", mask_index, "Guesses : ", words.index(token))

    # best_guess = ""
    # for j in list_of_list:
    #     best_guess = best_guess + " " + j[0]
    #
    # return best_guess


def main() -> None:
    index = Index.load(Config.index_file, Config.mapping_file)

    childhood_w_refs = " Presley's"  # gpt output

    embeddings = Embeddings(tokenizer, model).from_text(childhood_w_refs)
    print(embeddings)
    faiss.normalize_L2(embeddings)

    result_dists, result_ids = index.index.search(embeddings, 1)
    print("indexes:",result_ids,"result_ids dists:",result_dists,"\n\n")

    tokens = tokenizer.tokenize(childhood_w_refs) #
    print("tokens:",tokens,"\n\n") #

    for i,token in enumerate(tokens):
        source = index.get_source(int(result_ids[i]))
        text_from_section = get_page_section_from_wiki(source)  # последняя секция из вики с документа

        if "Ġ" in token:
            token = token.split("Ġ")[1]

        print("token:",token)
        new_text_from_section = text_from_section
        new_text_from_section_replaced = new_text_from_section.replace(token, "<mask>")

        token_ids = tokenizer.encode(new_text_from_section_replaced[:2100], return_tensors='pt')
        masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
        masked_pos = [mask.item() for mask in masked_position]  # позиции токенов в тексте на месте которых стоят <mask>

        print(get_prediction(new_text_from_section_replaced[:2100], token_ids, masked_pos, token))
        print("---------------------------------------------")



if __name__ == "__main__":
    main()
