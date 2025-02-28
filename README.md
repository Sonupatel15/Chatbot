# Building a Support Agent Chatbot for CDP  
## "How-to" Questions  

### Objective  
Develop a chatbot that can answer "how-to" questions related to four Customer Data Platforms (CDPs): **Segment, mParticle, Lytics, and Zeotap**. The chatbot should extract relevant information from the official documentation of these CDPs to guide users on performing tasks or achieving specific outcomes within each platform.  

---

## **Data Sources**  
The chatbot retrieves information from the following official documentation:  

- **Segment Documentation**: [https://segment.com/docs/?ref=nav](https://segment.com/docs/?ref=nav)  
- **mParticle Documentation**: [https://docs.mparticle.com/](https://docs.mparticle.com/)  
- **Lytics Documentation**: [https://docs.lytics.com/](https://docs.lytics.com/)  
- **Zeotap Documentation**: [https://docs.zeotap.com/home/en-us/](https://docs.zeotap.com/home/en-us/)  

---

## **Configuration Setup for the Project**  

### **Prerequisites**  
Ensure you have the following installed before running the chatbot:  

- Python 3.8+  
- Pip (Python package manager)  
- Virtual environment (optional but recommended)  
- Streamlit (for the frontend)  
- Requests and BeautifulSoup (for web scraping documentation)  

---

### **Installation Steps**  

Create a Virtual Environment
python -m venv venv
source venv/bin/activate

## **Install Required Dependencies**
pip install -r requirements.txt

## **Setting Up LM Studio**
Download and install LM Studio from https://lmstudio.ai/.
Load the NLP model.
Run the server to allow API-based responses for the chatbot.

## **Run the Server**

## **Running the Chatbot (Frontend using Streamlit)**
streamlit run app.py

## **Core Functionalities**

1. Answering "How-to" Questions
The chatbot can interpret and respond to user queries regarding performing tasks or using features in each CDP.

Example Queries:

Segment: "How do I set up a new source in Segment?"
mParticle: "How can I create a user profile in mParticle?"
Lytics: "How do I build an audience segment in Lytics?"
Zeotap: "How can I integrate my data with Zeotap?"


## **How do I set up a new source in Segment?**
![segment](https://github.com/user-attachments/assets/b9e2b37d-2bda-4175-8451-dc81b93984a3)
![segment_1](https://github.com/user-attachments/assets/4c601584-8c06-4077-9e15-1234f3dbc860)

## **How can I create a user profile in mParticle?**
![mparticle](https://github.com/user-attachments/assets/976d9f46-0243-455e-a43a-02baf4231232)
![mpartile_1](https://github.com/user-attachments/assets/344a8b7d-44fe-4778-8ad4-277b4c719f50)
![mparticle_3](https://github.com/user-attachments/assets/21d16198-c6db-4115-af36-7fa6b17c3199)

## **How do I build an audience segment in Lytics?**
![Lytics](https://github.com/user-attachments/assets/330c9c32-e9b1-4525-950c-99498fee378e)
![Lytics_1](https://github.com/user-attachments/assets/10b6543c-44c2-4d22-8b94-254e5bc1a75b)

## **How can I integrate my data with Zeotap?**
![Zeotap](https://github.com/user-attachments/assets/8df9ff78-36d2-4ca3-a09e-718136f3b685)
![Zeotap_1](https://github.com/user-attachments/assets/ddf67bbb-0b40-43ac-963d-60e8a0fec707)
![Zeotap_@](https://github.com/user-attachments/assets/8916e205-3ec7-4979-922e-9769ccc462e6)







