from torch.utils.data import DataLoader
from torchtune.models.llama3 import Llama3Tokenizer
from torchtune.datasets import instruct_dataset, alpaca_dataset, alpaca_inception_dataset

m_tokenizer = Llama3Tokenizer(path="/workspace/models/Meta-Llama-3.1-70B-Instruct/original/tokenizer.model")

datasets = {
    # "arbml/CIDAR": instruct_dataset(tokenizer=m_tokenizer, source="arbml/CIDAR",
    #                                 column_map={"input": "instruction", "output": "output"}),
    "akbargherbal/six_millions_instruction_dataset_for_arabic_llm_ft": alpaca_inception_dataset(tokenizer=m_tokenizer,
                                                                                                source="akbargherbal/six_millions_instruction_dataset_for_arabic_llm_ft"),
    # "akbargherbal/hadith_alpaca_ft": alpaca_dataset(tokenizer=m_tokenizer,
    #                                                 source="akbargherbal/hadith_alpaca_ft"),
    # "akbargherbal/ONE_MILLION_AR_TO_EN_SENTENCES_DATASET": alpaca_dataset(tokenizer=m_tokenizer,
    #                                                                       source="akbargherbal/ONE_MILLION_AR_TO_EN_SENTENCES_DATASET"),
    # "akbargherbal/10K_english_to_arabic_dataset_for_FT": alpaca_dataset(tokenizer=m_tokenizer,
    #                                                                     source="akbargherbal/10K_english_to_arabic_dataset_for_FT"),
    # "akbargherbal/10K_ARABIC_POEMS_FOR_FINETUNING": alpaca_dataset(tokenizer=m_tokenizer,
    #                                                                source="akbargherbal/10K_ARABIC_POEMS_FOR_FINETUNING",
    #                                                                column_map={"input": "instruction",
    #                                                                            "output": "output",
    #                                                                            "instruction": "system"}),
    "akbargherbal/cohere_msa_arabic_dataset": alpaca_inception_dataset(tokenizer=m_tokenizer,
                                                                       source="akbargherbal/cohere_msa_arabic_dataset"),
    # "akbargherbal/ONE_MILLION_EN_TO_AR_SENTENCES_DATASET": alpaca_dataset(tokenizer=m_tokenizer,
    #                                                                       source="akbargherbal/ONE_MILLION_EN_TO_AR_SENTENCES_DATASET"),
    # "AhmedBou/Arabic_instruction_dataset_for_llm_ft": alpaca_dataset(tokenizer=m_tokenizer,
    #                                                                  source="AhmedBou/Arabic_instruction_dataset_for_llm_ft"),

}


def main():
    for ds_name, ds in datasets.items():
        print(f"Dataset: {ds_name}")
        try:
            count = 0
            for batch in DataLoader(ds, batch_size=1, collate_fn=lambda x: x):
                count = count + len(batch)
                if count % 32768 == 0:
                    print(f"checked: {count}")
        except Exception as e:
            print(e)
            print(f"Error ds: {ds_name}")
            raise e


if __name__ == '__main__':
    main()
