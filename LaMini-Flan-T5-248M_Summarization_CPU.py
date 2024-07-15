#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
import streamlit as st
import PyPDF2
from docx import Document

#model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)
from langchain import PromptTemplate,  LLMChain


# In[ ]:





# In[ ]:


from huggingface_hub import notebook_login

notebook_login()


# In[ ]:


import transformers
from langchain import HuggingFacePipeline
pipeline = transformers.pipeline(
    'summarization',
    model = base_model,
    tokenizer = tokenizer,
    max_length = 500, 
    min_length = 10,)
from langchain import PromptTemplate,  LLMChain

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

template = """
              Create a summary highlighting my skillsets from this resume
              ```{text}```
           """

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)


# In[ ]:


#ext = """ I initially had trouble deciding between the paperwhite and the voyage because reviews more or less said the same thing: the paperwhite is great, but if you have spending money, go for the voyage.Fortunately, I had friends who owned each, so I ended up buying the paperwhite on this basis: both models now have 300 ppi, so the 80 dollar jump turns out pricey the voyage's page press isn't always sensitive, and if you are fine with a specific setting, you don't need auto light adjustment).It's been a week and I am loving my paperwhite, no regrets! The touch screen is receptive and easy to use, and I keep the light at a specific setting regardless of the time of day. (In any case, it's not hard to change the setting either, as you'll only be changing the light level at a certain time of day, not every now and then while reading).Also glad that I went for the international shipping option with Amazon. Extra expense, but delivery was on time, with tracking, and I didnt need to worry about customs, which I may have if I used a third party shipping service.
#Allow me to preface this with a little history. I am (was) a casual reader who owned a Nook Simple Touch from 2011. I've read the Harry Potter series, Girl with the Dragon Tattoo series, 1984, Brave New World, and a few other key titles. Fair to say my Nook did not get as much use as many others may have gotten from theirs.Fast forward to today. I have had a full week with my new Kindle Paperwhite and I have to admit, I'm in love. Not just with the Kindle, but with reading all over again! Now let me relate this review, love, and reading all back to the Kindle. The investment of 139.00 is in the experience you will receive when you buy a Kindle. You are not simply paying for a screen there is an entire experience included in buying from Amazon.I have been reading The Hunger Games trilogy and shall be moving onto the Divergent series soon after. Here is the thing with the Nook that hindered me for the past 4 years: I was never inspired to pick it up, get it into my hands, and just dive in. There was never that feeling of oh man, reading on this thing is so awesome. However, with my Paperwhite, I now have that feeling! That desire is back and I simply adore my Kindle. If you are considering purchasing one, stop thinking about it simply go for it. After a full week, 3 downloaded books, and a ton of reading, I still have half of my battery left as well.Make yourself happy. Inspire the reader inside of you.
#I am enjoying it so far. Great for reading. Had the original Fire since 2012. The Fire used to make my eyes hurt if I read too long. Haven't experienced that with the Paperwhite yet.
#I bought one of the first Paperwhites and have been very pleased with it its been a constant companion and I suppose Ive read, on average, a book every three days for the past however many years on it. I wouldnt give it up youd have to pry it from my cold dead fingers.For sundry logistical reasons, Ive also made good use of Amazons Kindle app on my iPhone. No Paperwhite screen, naturally, and all the cool usability that delivers, but it works well and has its own attractions as a companion to the Kindle.Of course, there are aspects of the Paperwhite which I would like to critique. Ah you knew that was coming somewhere, didnt you.As a member of BookBub, I get a daily list of alerts and book deals in my chosen genres. I take on many of them, however, Ive found that, even with the best will in the world, I cant keep up. Some days it seems that for every book I read, Ive bought two. Theres just so much good stuff out there! The accumulative effect of this is that the number of books actually on my Paperwhite has been creeping ever upward for some time. Its now at about 400.With this in mind, Ive noticed that while page-turning has remained exactly the same, just about every other action on the Kindle has become positively glacial. Not just very slow, but so slow you think its malfunctioning. The general consensus appears to be that its to be expected once one has that many books downloaded onto a Kindle, it will begin to behave in a flakey manner. This drives me mad. Amazon states it can hold thousands of books. I believe them. But I figure I would need a second Paperwhite to read while Im waiting for actions to complete on the first one.Read more
#I have to say upfront - I don't like coroporate, hermetically closed stuff like anything by Apple or in this case, Amazon. I like having devices on which I can put anything I want and use it. But...I was a fairly happy user of a Nook Touch for several years, but couldn't use all its functionalities since I live in Serbia. Then I lost the Nook and since no other devices can actually be fully used in Serbia (buying books with them, using their online capabilities) except the Kindle, and since no one except Amazon ships to Serbia, and since I've actually been a happy Amazon customer since 2005 over friends' accounts and since 2007 through my own, and since the Kindle definitely has the best technology - why not buy itSo I did. What I read in many reviews about the screen/light of the Paperwhite and similar devices was no problem with mine. The light disperses just fine, except a few black blotches (maybe you can see it in the picture) at the bottom of the screen, which are actually shadows of the black plastic casing and thus can't really be avoided. As you can see in the picture without the light - there are no blotches with light out.The Paperwhite's screen is just marvelous at 300 ppi, the touchscreen works just fine, the store works here in Serbia, and in these two days I've been using it, I'm a happy guy.I had to get the hang on how to make sideloaded books behave at least almost like Amazon books, but that's fine. That's the one thing I'd like to see Amazon do in some future upgrades: make the Kindle treat sideloaded books just like the ones bought from them directly, with sharing funcion (quotes and Goodreads) enabled and so on.The size is perfect, it sits very well in the hand, the light doesn't hurt the eyes in the dark (like the light on a tab does)... the packaging was fine, no problems there and what remains to be seen now is the battery life.So far, I can only recommend it.
#My previous kindle was a DX, this is my second kindle in years. Love the form factor and all but I do miss the physical buttons for page turning. There is a glitch in the software though. I use the English interface but occasionally I would like to translate words into traditional Chinese. However, it seems that the traditional Chinese characters cannot display correctly and become small boxes. The simplified Chinese characters can be displayed correctly though.
#Allow me to preface this with a little history. I am (was) a casual reader who owned a Nook Simple Touch from 2011. I've read the Harry Potter series, Girl with the Dragon Tattoo series, 1984, Brave New World, and a few other key titles. Fair to say my Nook did not get as much use as many others may have gotten from theirs.Fast forward to today. I have had a full week with my new Kindle Paperwhite and I have to admit, I'm in love. Not just with the Kindle, but with reading all over again! Now let me relate this review, love, and reading all back to the Kindle. The investment of 139.00 is in the experience you will receive when you buy a Kindle. You are not simply paying for a screen there is an entire experience included in buying from Amazon.I have been reading The Hunger Games trilogy and shall be moving onto the Divergent series soon after. Here is the thing with the Nook that hindered me for the past 4 years: I was never inspired to pick it up, get it into my hands, and just dive in. There was never that feeling of oh man, reading on this thing is so awesome. However, with my Paperwhite, I now have that feeling! That desire is back and I simply adore my Kindle. If you are considering purchasing one, stop thinking about it simply go for it. After a full week, 3 downloaded books, and a ton of reading, I still have half of my battery left as well.Make yourself happy. Inspire the reader inside of you.
#Just got mine right now. Looks the same as the previous generation except for the Kindle logo (it's black this time), feels a little heavier, the screen is a little warmer toned than the previous one, BUT the resolution is SO MUCH BETTER! The 300 ppi are definitely obvious and the new font is worth the money.Totally recommend it for a book lover!I'll give it 4 stars instead of 5 because I am used to a cooler screen, but I'm sure I'll get used to this one soon too :)
#I initially had trouble deciding between the paperwhite and the voyage because reviews more or less said the same thing: the paperwhite is great, but if you have spending money, go for the voyage.Fortunately, I had friends who owned each, so I ended up buying the paperwhite on this basis: both models now have 300 ppi, so the 80 dollar jump turns out pricey the voyage's page press isn't always sensitive, and if you are fine with a specific setting, you don't need auto light adjustment).It's been a week and I am loving my paperwhite, no regrets! The touch screen is receptive and easy to use, and I keep the light at a specific setting regardless of the time of day. (In any case, it's not hard to change the setting either, as you'll only be changing the light level at a certain time of day, not every now and then while reading).Also glad that I went for the international shipping option with Amazon. Extra expense, but delivery was on time, with tracking, and I didnt need to worry about customs, which I may have if I used a third party shipping service.
#I am enjoying it so far. Great for reading. Had the original Fire since 2012. The Fire used to make my eyes hurt if I read too long. Haven't experienced that with the Paperwhite yet.
#As reviewed by the wife This is the perfect thing for a new mommy who loves to read books! As soon as I had my baby girl, I had to stop reading my novels because I had to give the baby my full attention. But how much time do I spend laying in bed, on my side, with the baby while she feeds Especially prior to 4 months when feeding took as long as 45 minutes! Not to mention cluster feeds during growth spurts! A book was out of the question because it's sooo tiring to hold up in side-lying-breastfeeding-position. My new Kindle was the solution! It's light, easy to disinfect (I use baby wipes on the case whenever I feel like it's not clean enough to go on the bed with the baby) and you don't need to keep a finger in the middle of the pages to keep it from closing - you know what I mean!And for some reason it's much easier to go through a book with a Kindle compared to an actual book. I never thought I'd be converted into the Kindle culture but here we are!
#"""


# In[ ]:


#print(llm_chain.run(text))


# In[ ]:


# Streamlit app title
st.title("PDF and Word Document Reader")

# Upload file via Streamlit
file = st.file_uploader("Upload a PDF or Word Document", type=["pdf", "docx"])

if file:
    file_type = file.type
    if file_type == "application/pdf":
        # Read PDF file
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            print(page_num)
            page = pdf_reader.pages[page_num]
            #page = pdf_reader.pages[1]
            text += page.extract_text()

    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Read Word document
        doc = Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text

    # Display the content
    st.subheader("Skill Set:")
    #st.text(text)
    final=llm_chain.run(text)
    st.text(final)
    
