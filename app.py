import pandas as pd
from fuzzywuzzy import process, fuzz
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet as wn
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re
import csv
import asyncio
from aiohttp import ClientSession
from io import StringIO
from datetime import datetime
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# nltk.download('vader_lexicon')
# nltk.download('wordnet')

sia = SentimentIntensityAnalyzer()
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

df = pd.read_csv("data.csv")
df_timetable = pd.read_csv("timetable.csv")
df_events = pd.read_csv("events.csv")
df_highschool_info = pd.read_csv('highschool_info.csv')
df_teacher = pd.read_csv("teachers.csv")

df.dropna(inplace=True, how='all')
df_timetable.dropna(inplace=True, how='all')
df_events.dropna(inplace=True, how='all')
df_highschool_info.dropna(inplace=True, how='all')
df_teacher.dropna(inplace=True, how='all')


df.columns = df.columns.str.lower().str.replace(' ', '_')
df_timetable.columns = df_timetable.columns.str.lower().str.replace(' ', '_')
df_events.columns = df_events.columns.str.lower().str.replace(' ', '_')
df_highschool_info.columns = df_highschool_info.columns.str.lower().str.replace(' ', '_')
df_teacher.columns = df_teacher.columns.str.lower().str.replace(' ', '_')

async def preprocess_query(query):
    return query.lower()

async def is_multi_word_query(query):
    return len(query.split()) > 1

async def get_highschool_info(user_input):
    user_input = await preprocess_query(user_input)
    response = ""
    
    if "presentation" in user_input or "info" in user_input or "information" in user_input:
        if 'presentation' in df_highschool_info.columns:
            response = df_highschool_info['presentation'].dropna().iloc[0]
        else:
            response = "Presentation information is not available."
    elif "history" in user_input:
        if 'history' in df_highschool_info.columns:
            history_section = df_highschool_info['history'].dropna().iloc[0]
            response = (
                "The 'Mihai Viteazul' National College in Bucharest was founded in 1865 as a subdivision "
                "of the Saint Sava High School. It became autonomous two years later and functioned in "
                "various locations until the end of World War I. Construction of the current headquarters "
                "began in 1921 and was completed after four years. The school features various facilities "
                "including classrooms, laboratories, a library, a dormitory, an auditorium, a chapel, an "
                "amphitheater, and a gym. Over the years, it has continued to be a significant educational "
                "and cultural institution, producing many notable alumni."
            )
        else:
            response = "History information is not available."
    elif "results" in user_input or "rank" in user_input or "ranking" in user_input:
        if 'results' in df_highschool_info.columns:
            response = df_highschool_info['results'].dropna().iloc[0]
        else:
            response = "Results information is not available."
    elif "notable alumni" in user_input or "famous" in user_input or "alumni" in user_input:
        if 'history' in df_highschool_info.columns:
            notable_alumni_section = df_highschool_info['history'].dropna().iloc[0]
            notable_alumni_marker = "Notable alumni include:"
            if notable_alumni_marker in notable_alumni_section:
                alumni_list = notable_alumni_section.split(notable_alumni_marker)[1].strip()
                response = f"Notable alumni include:\n{alumni_list}"
            else:
                response = "Notable alumni information is not available."
        else:
            response = "Notable alumni information is not available."
    else:
        response = "I'm sorry, I couldn't find any information related to your query."
    return response

async def list_all_events(user_input):
    user_input = await preprocess_query(user_input)
    response = ""
    event_info = df_events[['name', 'timeline']]
    event_info_str = event_info.to_string(index=False, header=False)
    # Split each line and add comma between event name and date
    lines = event_info_str.split('\n')
    formatted_lines = [f"{line.split(maxsplit=1)[0]} {line.split(maxsplit=1)[1]}" for line in lines]
    # Join lines with an empty line between each row
    response += '\n'.join(formatted_lines)
    return response

async def describe_all_events(user_input):
    user_input = await preprocess_query(user_input)
    response = ""
    event_info = df_events[['name', 'timeline', 'description']]

    event_info_str = event_info.to_string(index=False, header=False)
    lines = event_info_str.split('\n')
    formatted_lines = [f"{line.split(maxsplit=1)[0]} {line.split(maxsplit=1)[1]}" for line in lines]
    
    for i, line in enumerate(formatted_lines):
        description = event_info.iloc[i]['description']
        formatted_lines[i] += f"\nDescription: {description}"

    response += '\n\n'.join(formatted_lines)
    return response

async def search_teachers_by_subject(filename, subject):
    async def read_csv(filename):
        data = {}
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                subj = row['Subject'].lower()
                if subj in data:
                    data[subj].append(row)
                else:
                    data[subj] = [row]
        return data

    data = await read_csv(filename)

    if subject.lower().startswith("who teaches"):
        subject = subject.split("who teaches", 1)[-1].strip()

    teachers = data.get(subject.lower(), [])
    if teachers:
        return [f"{teacher['Name']} ({teacher['Email']}) Severity level: {teacher['Severity']}" for teacher in teachers]
    else:
        return []
    
async def get_teachers_with_strict_score(file_path, strict_score):
    result = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                severity = int(row['Severity'])
            except ValueError:
                continue
            if severity == strict_score:
                result.append((row['Name'], row['Subject']))
    return result

# async def list_all_teachers(file_path):
#     teachers_info = []
#     with open(file_path, mode='r', encoding='utf-8') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             name = row['Name']
#             subject = row['Subject']
#             email = row['Email']
#             teachers_info.append(f"{name}, Subject: {subject}, Email adress: {email}")
#     return teachers_info

# async def all_teachers():
#     teachers_info = await list_all_teachers('teachers.csv')
#     for teacher_info in teachers_info:
#         print(teacher_info)

async def list_all_teachers():
    teachers_info = []
    df_teacher = pd.read_csv('teachers.csv')  
    for index, row in df_teacher.iterrows():  
        name = row['Name']
        subject = row['Subject']
        email = row['Email']
        teachers_info.append(f"\n{name} - Subject: {subject} - Email address: {email}")
    return teachers_info


async def query_robotics_clubs(df):
    robotics_keywords = ['robotics', 'robots']
    relevant_clubs = df[df['description'].str.contains('|'.join(robotics_keywords), case=False, na=False)]
    relevant_clubs = relevant_clubs[relevant_clubs['clubname'].isin(['Qube', 'Ignite', 'Neurobotix'])]

    clubs_string = relevant_clubs[['description']].to_string(index=False, header=False)
    clubs_lines = clubs_string.split('\n')

    cleaned_lines = [line.strip() for line in clubs_lines if line.strip()]

    clubs_with_space = '\n'.join(cleaned_lines)
    return clubs_with_space

async def query_dataframe(query, df):
    query = await preprocess_query(query)
    multi_word = await is_multi_word_query(query)
    matches = []

    for col in df.columns:
        if multi_word:
            match = process.extractOne(query, df[col].astype(str).values, scorer=fuzz.partial_ratio)
        else:
            query_tokens = query.split()
            match = max((process.extractOne(token, df[col].astype(str).values, scorer=fuzz.partial_ratio) for token in query_tokens), key=lambda x: x[1] if x else 0)

        if match and match[1] > 80:
            matches.append((col, match))

    if matches:
        if multi_word:
            filtered_df = df[df.apply(lambda row: any(fuzz.partial_ratio(query, str(row[col])) > 80 for col, _ in matches), axis=1)]
        else:
            filtered_df = df[df.apply(lambda row: all(any(fuzz.partial_ratio(token, str(row[col])) > 80 for col, _ in matches) for token in query.split()), axis=1)]
        
        if not filtered_df.empty:
            return filtered_df

    return pd.DataFrame()

async def generate_class_schedule(class_name, df_timetable):
    if class_name.lower() in df_timetable.columns:
        return df_timetable[class_name.lower()].dropna().tolist()
    else:
        return []
    
async def get_class_timetable(user_input, df_timetable):
    if any(keyword in user_input.lower() for keyword in ['class', 'classes', 'scheduel','schedgual','timetable' ]):
        pattern = r'(?<!\d)(9[A-I]|10[A-J]|11[A-J]|12[A-J])(?!\d)'
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            class_identifier = match.group(0).upper()
            class_club_row = df_timetable[df_timetable['class'].str.contains(class_identifier, case=False, na=False)]
            
            if not class_club_row.empty:
                class_club_row_str = class_club_row.to_string(index=False, header=False, na_rep='').strip()                
                class_club_row_cleaned = '\n'.join(line.strip() for line in class_club_row_str.split('\n') if line.strip())
                return f"Class {class_identifier}:\n{class_club_row_cleaned}"
            else:
                return f"No timetable found for class {class_identifier}."
        else:
            return "No valid class identifier found in the input."
    return "No relevant keywords found in the input."

async def get_club_info_email(club_keyword, df):
    club_row = df[df['clubname'].str.lower().str.contains(club_keyword, case=False)]
    if not club_row.empty:
        if 'email' in df.columns:
            club_info_list = []
            for _, row in club_row.iterrows():
                email = row['email'] if pd.notna(row['email']) else "Not available"
                club_info_list.append(f"Club: {row['clubname']}\nEmail: {email}")
            club_info_text = "\n\n".join(club_info_list)
        else:
            club_info_list = [f"Club: {row['clubname']}\nEmail: Not available" for _, row in club_row.iterrows()]
            club_info_text = "\n\n".join(club_info_list)

        return club_info_text
    else:
        return f"I'm sorry, I couldn't find any information about the {club_keyword} club."

async def get_club_info(club_keyword, df):
    club_row = df[df['clubname'].str.lower().str.contains(club_keyword, case=False)]
    if not club_row.empty:
        club_info = club_row.to_string(index=False, header=False, na_rep='').strip()
        return '\n'.join(line.strip() for line in club_info.split('\n') if line.strip())
    else:
        return f"I'm sorry, I couldn't find any information about the {club_keyword} club."

async def generate_response(user_input, df):
    user_input = await preprocess_query(user_input)

    sentiment = sia.polarity_scores(user_input)
    if sentiment['compound'] < -0.3:
        return "I understand you're upset. How can I assist you with your concerns?"

    ner_results = ner_pipeline(user_input)
    entities = [res['word'] for res in ner_results if res['entity'].startswith('B')]

    if "list clubs" in user_input or "all clubs" in user_input  or "clubs" in user_input:
        if 'clubname' in df.columns:
            clubs_list = df['clubname'].unique()
            response_text = "Here is a list of all the clubs:\n\n"
            response_text += '\n'.join(clubs_list)
        else:
            response_text = "The 'clubname' column is not found in the dataframe."
    elif 'orpheus' in user_input or 'theater' in user_input:
        if 'contact' in user_input or 'email' in user_input or 'join' in user_input:
            response_text = await get_club_info_email('orpheus', df)
        else:
            response_text = await get_club_info('orpheus', df)
    elif 'photography' in user_input or 'photo' in user_input:
        if 'contact' in user_input or 'email' in user_input or 'join' in user_input:
            response_text = await get_club_info_email('capture', df)
        else:
            response_text = await get_club_info('capture', df)
    elif 'physics' in user_input or 'aerospace' in user_input:
        if 'contact' in user_input or 'email' in user_input or 'join' in user_input:
            response_text = await get_club_info_email('physics', df)
        else:
            response_text = await get_club_info('physics', df)
    elif 'literary' in user_input or 'cafe' in user_input:
        if 'contact' in user_input or 'email' in user_input or 'join' in user_input:
            response_text = await get_club_info_email('literary', df)
        else:
            response_text = await get_club_info('literary', df)
    elif 'interact' in user_input:
        if 'contact' in user_input or 'email' in user_input or 'join' in user_input:
            response_text = await get_club_info_email('interact', df)
        else:
            response_text = await get_club_info('interact', df)
    elif 'exchange' in user_input or 'foreign' in user_input:
        if 'contact' in user_input or 'email' in user_input or 'join' in user_input:
            response_text = await get_club_info_email('exchange', df)
        else:
            response_text = await get_club_info('exchange', df)
    elif 'film' in user_input or 'movie' in user_input:
        if 'contact' in user_input or 'email' in user_input or 'join' in user_input:
            response_text = await get_club_info_email('film', df)
        else:
            response_text = await get_club_info('film', df)
    elif 'floare de colt' in user_input or 'ecology' in user_input or 'ecological' in user_input:
        if 'contact' in user_input or 'email' in user_input or 'join' in user_input:
            response_text = await get_club_info_email('floare de colt', df)
        else:
            response_text = await get_club_info('floare de colt', df)
    elif 'radio' in user_input or 'radio club' in user_input or 'video audio' in user_input:
        if 'contact' in user_input or 'email' in user_input or 'join' in user_input:
            response_text = await get_club_info_email('radio', df)
        else:
            response_text = await get_club_info('radio', df)
    elif 'mun' in user_input or 'munob' in user_input:
        if 'contact' in user_input or 'email' in user_input or 'join' in user_input:
            response_text = await get_club_info_email('united nations', df)
        else:
            response_text = await get_club_info('united nations', df)
    elif 'debate' in user_input or 'public speaking' in user_input:
        if 'contact' in user_input or 'email' in user_input or 'join' in user_input:
            response_text = await get_club_info_email('debate', df)
        else:
            response_text = await get_club_info('debate', df)
    elif 'robotics' in user_input or 'robots' in user_input:
        robotics_clubs = await query_robotics_clubs(df)
        response_text = "Here is information about robotics clubs:\n\n" + robotics_clubs if robotics_clubs else "No information found about robotics clubs."
    elif 'neurobotics' in user_input or 'neurobotix' in user_input or 'neuro' in user_input:
        if 'contact' in user_input or 'email' in user_input or 'join' in user_input:
            response_text = await get_club_info_email('Neurobotix', df)
        else:
            response_text = await get_club_info('Neurobotix', df)
    elif 'qube' in user_input or 'qube.' in user_input:
        if 'contact' in user_input or 'email' in user_input or 'join' in user_input:
            response_text = await get_club_info_email('Qube', df)
        else:
            response_text = await get_club_info('Qube', df)
    elif 'ignite' in user_input or 'ignite.' in user_input:
        if 'contact' in user_input or 'email' in user_input or 'join' in user_input:
            response_text = await get_club_info_email('ignite', df)
        else:
            response_text = await get_club_info('ignite', df)
    elif 'sever' in user_input or 'strict' in user_input or 'strictest' in user_input:
        if "most" in user_input:
            teachers_with_strict_score_5 = await get_teachers_with_strict_score('teachers.csv', 5)
            if teachers_with_strict_score_5:
                severest_teachers = [f"{teacher[0]}, Subject: {teacher[1]}, Severity score of: 5" for teacher in teachers_with_strict_score_5]
                return "\n".join(severest_teachers)
            else:
                return "There are no teachers with a severity score of 5."
        elif "least sever" in user_input:
            teachers_with_strict_score_5 = await get_teachers_with_strict_score('teachers.csv', 1)
            if teachers_with_strict_score_5:
                severest_teachers = [f"{teacher[0]}, Subject: {teacher[1]}, Severity score of: 1" for teacher in teachers_with_strict_score_5]
                return "\n".join(severest_teachers)
            else:
                return "There are no teachers with a severity score of 1."
        else:
            teachers_with_strict_score_5 = await get_teachers_with_strict_score('teachers.csv', 5)
            if teachers_with_strict_score_5:
                severest_teachers = [f"{teacher[0]}, Subject: {teacher[1]}, Severity score of: 5" for teacher in teachers_with_strict_score_5]
                return "\n".join(severest_teachers)
            else:
                return "There are no teachers with a severity score of 5."
    elif 'teacher' in user_input or 'teaches' in user_input or 'teach' in user_input:
        if 'all' in user_input or 'list' in user_input:
            response_text = await list_all_teachers()
        else: 
            subject = re.findall(r'(?i)\b(?:teach|teaches|teaching)\b\s*(\w+)', user_input)
            if subject:
                subject = subject[0]
                teachers = await search_teachers_by_subject("teachers.csv", subject)
                if teachers:
                    response_text = f"Here are the teachers who teach {subject} and their email addresses:\n\n"
                    response_text += "\n".join(teachers)
                else:
                    response_text = f"No teachers found for the subject: {subject}"
            else:
                response_text = "Please mention the subject along with the teacher query."
    elif 'upcoming' in user_input or 'recent' in user_input or 'events' in user_input:
        if 'list' in user_input or 'all' in user_input:
            response_text = await list_all_events(user_input)
        else:
            response_text = await describe_all_events(user_input)
    elif 'highschool' in user_input or 'school' in user_input or 'info' in user_input or 'history' in user_input or 'information' in user_input:
        response_text = await get_highschool_info(user_input)
    elif any(keyword in user_input.lower() for keyword in ['class', 'classes', 'scheduel','schedgual','timetable']):
        response_text = await get_class_timetable(user_input, df_timetable)
    else:
        results = await query_dataframe(user_input, df)
        if not results.empty:
            response_text = "Here are the matching results from the database:\n\n"
            response_text += '\n'.join([' | '.join(map(str, row.dropna())) for row in results.values])
        else:
            class_schedule = await generate_class_schedule(user_input, df_timetable)
            response_text = f"Here is the schedule for class {user_input}:\n\n" + '\n'.join(class_schedule) if class_schedule else "I couldn't find any information related to your query."

    return response_text

# async def chat_with_csv(df):
#     print("Welcome to Paddington, the School's chatbot! Type 'exit' to end the conversation. What can I help you with?")

#     while True:
#         user_input = input("Student: ")
#         if user_input.lower() == 'exit':
#             break

#         response = await generate_response(user_input, df)
#         print(f"Paddington: {response}")

# # To run the async function
# asyncio.run(chat_with_csv(df))

@app.route('/')
def index():
    return render_template('index.html')



# Flask route to handle chat with CSV
@app.route('/chat', methods=['POST'])
async def chat_with_csv():    
    data = request.json
    user_input = data.get('query', '')
    response = await generate_response(user_input, df)
    return jsonify({'response': response})

@app.route('/your-endpoint', methods=['POST'])
def your_endpoint():
    # Process the request and generate the response
    response_data = {"message": "Your response message"}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
