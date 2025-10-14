import os

cur_dir=os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(cur_dir,"plain_prompt.txt",'r',encoding='utf-8')) as f:
    PlainPrompt_User_Template="".join(f.readlines())