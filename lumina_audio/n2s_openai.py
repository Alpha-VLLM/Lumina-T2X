import argparse
import os

from openai import OpenAI
import pandas as pd
import requests

openai_key = "your openai api key here"
base_url = ""


def get_struct(caption):
    if base_url != "":
        client = OpenAI(api_key=openai_key, base_url=base_url)
    else:
        client = OpenAI(api_key=openai_key)

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"I want to know what sound might be in the given scene and you need to give me the results in the following format:\
                Question: A bird sings on the river in the morning, a cow passes by and scares away the bird.\
                Answer: <running water& all>@<birds chriping& start>@<cow footsteps& mid>@<birds flying away& end>.\
                Question: cellphone ringing a variety of tones followed by a loud explosion and fire crackling as a truck engine runs idle\
                Answer: <variety cellphone ringing tones& start>@<loud explosion& end>@<fire crackling& end>@<truck engine idle& end>\
                Question: Train passing followed by short honks three times \
                Answer: <train passing& all>@<short honks three times& end>\
                All indicates the sound exists in the whole scene \
                Start, mid, end indicates the time period the sound appear.\
                Question: {caption} \
                Answer:",
            },
        ],
        temperature=0.0,
    )

    return completion.choices[0].message.content


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tsv_path = args.tsv_path
    ori_df = pd.read_csv(tsv_path, sep="\t")
    index = 0
    end = len(ori_df)
    name = os.path.basename(tsv_path)[:-4]
    f = open(f"{name}.txt", "w")
    newcap_list = []
    while index < end - 1:
        try:
            df = ori_df.iloc[index:end]
            for t in df.itertuples():
                index = int(t[0])
                ori_caption = getattr(t, "caption")
                strcut_cap = get_struct(ori_caption)
                if "sorry" in strcut_cap.lower():
                    strcut_cap = f"<{ori_caption.lower()}, all>"
                newcap_list.append(strcut_cap)
                f.write(f"{index}\t{strcut_cap}\n")
                f.flush()
        except:
            print("error")
            f.flush()
    f.close()
    with open(f"{name}.txt") as f:
        lines = f.readlines()
    id2cap = {}
    for line in lines:
        index, caption = line.strip().split("\t")
        id2cap[int(index)] = caption

    df = pd.read_csv(f"{name}.tsv", sep="\t")
    df["struct_cap"] = df.index.map(id2cap)
    df.to_csv(f"{name}_struct.tsv", sep="\t", index=False)
