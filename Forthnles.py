import requests
import pandas as pd
import numpy as np
import os
import json
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.model_selection import train_test_split
from fake_useragent import UserAgent
from dotenv import load_dotenv
# Load environment variables if any
load_dotenv()
class ChatAIDatasetGenerator:
    def __init__(self, output_dir="chat_datasets"):
        """
        Initialize the Chat AI Dataset Generator
        
        Args:
            output_dir (str): Directory to save generated datasets
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Setup browser options for web scraping
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.user_agent = UserAgent()
        self.chrome_options.add_argument(f"user-agent={self.user_agent.random}")
    
    def web_search(self, query, num_results=10):
        """
        Generate synthetic search results (removed dependency on external search APIs)
        
        Args:
            query (str): Search query
            num_results (int): Number of results to return
            
        Returns:
            list: List of synthetic search results
        """
        try:
            # Generate synthetic search results instead of using DuckDuckGo
            print(f"Generating synthetic search results for: {query}")
            
            # Extract keywords from query
            keywords = query.lower().split()
            keywords = [word for word in keywords if len(word) > 3]
            
            # Expanded list of domain names (over 200)
            domains = [
                # Common TLDs
                "example.com", "information.org", "knowledge.net", "data.io", "research.edu", 
                "resources.co", "learning.site", "reference.info", "guides.tech", "academy.online",
                "insights.app", "tutorials.dev", "library.blog", "archive.digital", "center.network",
                "database.cloud", "repository.co.uk", "directory.ca", "index.store", "catalog.xyz",
                "portal.world", "access.site", "source.space", "channel.live", "platform.media",
                "hub.zone", "nexus.pro", "stack.tech", "compass.guide", "pipeline.tools",
                "forge.works", "matrix.systems", "cluster.solutions", "fabric.network", "grid.services",
                "sphere.global", "cube.studio", "pulse.agency", "circuit.design", "spectrum.life",
                "galaxy.club", "wisdom.university", "intellect.school", "scholar.institute", "genius.academy",
                
                # Academic domains
                "harvard.edu", "stanford.edu", "mit.edu", "berkeley.edu", "oxford.ac.uk",
                "princeton.edu", "yale.edu", "caltech.edu", "columbia.edu", "cambridge.ac.uk",
                "chicago.edu", "cornell.edu", "upenn.edu", "nyu.edu", "umich.edu",
                "ucla.edu", "ucsb.edu", "ucsd.edu", "ucdavis.edu", "purdue.edu",
                
                # Tech/Information domains
                "techinfo.dev", "codebase.io", "devhub.net", "aiprogress.tech", "datastream.org",
                "machinelearning.ai", "techstack.dev", "datascience.org", "researchgate.net", "webresource.io",
                "openai.com", "tensorflow.org", "pytorch.org", "kaggle.com", "github.io",
                "stackoverflow.com", "medium.com", "dev.to", "hackernoon.com", "infoworld.com",
                "wired.com", "theverge.com", "techcrunch.com", "arstechnica.com", "zdnet.com",
                
                # Business/Professional domains
                "linkedin.com", "indeed.com", "glassdoor.com", "monster.com", "forbes.com",
                "bloomberg.com", "wsj.com", "entrepreneur.com", "fastcompany.com", "inc.com",
                "business.com", "businessinsider.com", "cnbc.com", "economist.com", "hbr.org",
                "mckinsey.com", "deloitte.com", "accenture.com", "pwc.com", "ey.com",
                
                # Health/Science domains
                "nih.gov", "who.int", "mayoclinic.org", "webmd.com", "healthline.com",
                "medlineplus.gov", "medicalnews.com", "science.org", "nature.com", "pnas.org",
                "sciencedirect.com", "ieee.org", "acm.org", "cell.com", "nejm.org",
                "jama.com", "lancet.com", "bmj.com", "sciencemag.org", "plos.org",
                
                # News/Media domains
                "nytimes.com", "washingtonpost.com", "bbc.com", "reuters.com", "apnews.com",
                "cnn.com", "foxnews.com", "nbcnews.com", "abcnews.go.com", "cbsnews.com",
                "usatoday.com", "latimes.com", "chicagotribune.com", "bostonglobe.com", "guardian.co.uk",
                "time.com", "newsweek.com", "economist.com", "ft.com", "aljazeera.com",
                
                # Reference domains
                "wikipedia.org", "britannica.com", "dictionary.com", "merriam-webster.com", "thesaurus.com",
                "howstuffworks.com", "smithsonianmag.com", "nationalgeographic.com", "discovery.com", "history.com",
                "pbs.org", "archive.org", "loc.gov", "gutenberg.org", "worldcat.org",
                
                # Social/Community domains
                "reddit.com", "quora.com", "stackexchange.com", "discourse.org", "forum.com",
                "community.dev", "discuss.io", "tribe.so", "circle.so", "slack.com",
                "discord.com", "gitter.im", "telegram.org", "signal.org", "whatsapp.com",
                
                # Government domains
                "usa.gov", "whitehouse.gov", "senate.gov", "house.gov", "state.gov",
                "defense.gov", "nasa.gov", "epa.gov", "fda.gov", "cdc.gov",
                "irs.gov", "fbi.gov", "justice.gov", "energy.gov", "ed.gov",
                
                # Local/Regional domains
                "citydata.org", "regional.info", "localguide.net", "mycity.com", "townhall.gov",
                "countyinfo.org", "stateguide.com", "regionaldata.net", "localresources.org", "communityhub.info",
                
                # Additional variety domains
                "dataquest.io", "analyticsvidhya.com", "towardsdatascience.com", "kdnuggets.com", "machinelearningmastery.com",
                "datasciencecentral.com", "analyticsinsight.net", "learndatasci.com", "datacamp.com", "cognitiveclass.ai",
                "digitalocean.com", "freecodecamp.org", "codecademy.com", "w3schools.com", "tutorialspoint.com",
                "geeksforgeeks.org", "javatpoint.com", "guru99.com", "learnpython.org", "realpython.com"
            ]
            
            results = []
            for i in range(num_results):
                # Create synthetic result
                domain = np.random.choice(domains)
                # Use keywords in the URL and snippet
                keyword = np.random.choice(keywords) if keywords else "topic"
                
                url = f"https://{domain}/{keyword}-{i+1}"
                snippet = f"This page contains information about {query}. Find detailed resources, examples, and references about {keyword}."
                
                results.append({
                    "url": url,
                    "snippet": snippet
                })
            
            return results
        except Exception as e:
            print(f"Error generating synthetic search results: {e}")
            # Return fallback data
            return [{"url": f"https://example.com/sample{i}", "snippet": f"Sample data for {query} #{i}"} for i in range(num_results)]
    
    def scrape_webpage(self, url):
        """
        Generate synthetic webpage content instead of scraping
        
        Args:
            url (str): URL to process
            
        Returns:
            dict: Dictionary containing synthetic content
        """
        try:
            # Instead of actual web scraping, generate synthetic content
            print(f"Generating synthetic content for: {url}")
            
            # Extract keywords from URL
            path_parts = url.split('/')
            keywords = []
            for part in path_parts:
                if part and part not in ["https:", "http:", "www."]:
                    # Extract words from domain or path
                    words = part.replace('.com', '').replace('.org', '').replace('.net', '').split('-')
                    keywords.extend([w for w in words if len(w) > 2])
            
            # Generate synthetic content
            title = f"Information about {' '.join(keywords[:2]).title()}"
            
            # Generate paragraphs
            paragraphs = []
            for i in range(5):
                paragraph = f"This is paragraph {i+1} about {' '.join(keywords)}. "
                paragraph += "It contains information that can be used for training conversational AI models. "
                paragraph += f"The topic of {keywords[0] if keywords else 'this content'} is important for understanding various aspects of the subject. "
                paragraph += f"There are multiple approaches and methodologies related to {keywords[-1] if len(keywords) > 1 else 'this area'}."
                paragraphs.append(paragraph)
            
            # Generate headings
            headings = [
                f"Introduction to {' '.join(keywords[:2]).title()}",
                f"Understanding {keywords[0].title() if keywords else 'The Topic'}",
                f"Applications of {keywords[-1].title() if len(keywords) > 1 else 'This Subject'}"
            ]
            
            # Generate QA pairs
            qa_pairs = []
            qa_templates = [
                {"question": f"What is {keywords[0] if keywords else 'this topic'}?", 
                 "answer": f"{keywords[0].title() if keywords else 'This topic'} refers to a set of methods and techniques used in various contexts."},
                {"question": f"How can I learn more about {' '.join(keywords[:2]) if keywords else 'this subject'}?", 
                 "answer": f"There are many resources available for learning about {' '.join(keywords[:2]) if keywords else 'this subject'}, including online courses, books, and tutorials."},
                {"question": f"What are the benefits of {keywords[-1] if len(keywords) > 1 else 'this approach'}?", 
                 "answer": f"The benefits of {keywords[-1] if len(keywords) > 1 else 'this approach'} include improved efficiency, better outcomes, and broader applications."}
            ]
            
            # Add more variety to QA pairs
            for i in range(min(5, len(keywords))):
                if i < len(keywords):
                    q = f"Can you explain how {keywords[i]} works in practice?"
                    a = f"{keywords[i].title()} works through a series of steps that involve data processing, analysis, and application."
                    qa_pairs.append({"question": q, "answer": a})
            
            qa_pairs.extend(qa_templates)
            
            return {
                "url": url,
                "title": title,
                "paragraphs": paragraphs,
                "headings": headings,
                "qa_pairs": qa_pairs,
                "full_text": " ".join(paragraphs)
            }
        except Exception as e:
            print(f"Error generating content for {url}: {e}")
            return {"url": url, "error": str(e)}
    
    def generate_conversation_dataset(self, conversation_types, samples_per_type=50):
        """
        Generate a dataset of conversation examples
        
        Args:
            conversation_types (list): List of conversation types/themes
            samples_per_type (int): Number of samples per conversation type
            
        Returns:
            list: List of conversation examples
        """
        conversations = []
        
        for conv_type in conversation_types:
            print(f"Generating conversations for type: {conv_type}")
            
            # Search for relevant content
            search_results = self.web_search(f"{conv_type} conversation examples", num_results=min(10, samples_per_type // 2))
            
            type_conversations = []
            
            for result in search_results:
                scraped_content = self.scrape_webpage(result["url"])
                if "error" not in scraped_content:
                    # Extract QA pairs if found
                    if scraped_content.get("qa_pairs"):
                        for qa_pair in scraped_content["qa_pairs"]:
                            conversation = {
                                "conversation_type": conv_type,
                                "messages": [
                                    {"role": "user", "content": qa_pair["question"]},
                                    {"role": "assistant", "content": qa_pair["answer"]}
                                ]
                            }
                            type_conversations.append(conversation)
                    
                    # Generate synthetic conversations from paragraphs
                    if scraped_content.get("paragraphs") and len(type_conversations) < samples_per_type:
                        for paragraph in scraped_content["paragraphs"]:
                            if len(paragraph) > 100:
                                # Create a synthetic conversation
                                sentences = paragraph.split('.')
                                if len(sentences) >= 3:
                                    # Create user question from content
                                    topic_words = [w for w in conv_type.split() if len(w) > 3]
                                    if topic_words:
                                        topic_word = np.random.choice(topic_words)
                                        user_questions = [
                                            f"Can you tell me about {topic_word}?",
                                            f"What's your opinion on {topic_word}?",
                                            f"How does {topic_word} work?",
                                            f"I'm interested in learning about {topic_word}. Can you help?",
                                            f"What should I know about {topic_word}?"
                                        ]
                                        user_message = np.random.choice(user_questions)
                                        
                                        conversation = {
                                            "conversation_type": conv_type,
                                            "messages": [
                                                {"role": "user", "content": user_message},
                                                {"role": "assistant", "content": paragraph}
                                            ]
                                        }
                                        type_conversations.append(conversation)
                
                if len(type_conversations) >= samples_per_type:
                    break
            
            # If we don't have enough conversations, create synthetic ones
            while len(type_conversations) < samples_per_type:
                # Generate a synthetic conversation based on topic
                topic_words = [w for w in conv_type.split() if len(w) > 3]
                if not topic_words:
                    topic_words = [conv_type]
                
                topic_word = np.random.choice(topic_words)
                
                user_questions = [
                    f"Can you tell me about {topic_word}?",
                    f"What's your opinion on {topic_word}?",
                    f"How does {topic_word} work?",
                    f"I'm interested in learning about {topic_word}. Can you help?",
                    f"What should I know about {topic_word}?"
                ]
                
                assistant_responses = [
                    f"I'd be happy to discuss {topic_word}. It's an interesting topic that involves several key aspects.",
                    f"When it comes to {topic_word}, there are multiple perspectives to consider.",
                    f"{topic_word} is a fascinating subject. Let me share some information about it.",
                    f"I understand you're interested in {topic_word}. Here's what I know about it.",
                    f"Thanks for asking about {topic_word}. Let me provide some insights."
                ]
                
                conversation = {
                    "conversation_type": conv_type,
                    "messages": [
                        {"role": "user", "content": np.random.choice(user_questions)},
                        {"role": "assistant", "content": np.random.choice(assistant_responses)}
                    ]
                }
                
                # Add follow-up exchanges (25% chance)
                if np.random.random() < 0.25:
                    follow_up_questions = [
                        "Can you tell me more?",
                        "That's interesting. What else should I know?",
                        "Do you have any specific examples?",
                        "How does this compare to other approaches?",
                        "What are the advantages and disadvantages?"
                    ]
                    
                    follow_up_responses = [
                        f"Certainly! When it comes to {topic_word}, there are additional considerations.",
                        f"I'd be happy to elaborate. Another important aspect of {topic_word} is...",
                        f"Of course! Here's some more information about {topic_word}.",
                        f"Great question. Regarding {topic_word}, experts often highlight...",
                        f"Absolutely. Let me provide more context about {topic_word}."
                    ]
                    
                    conversation["messages"].append({"role": "user", "content": np.random.choice(follow_up_questions)})
                    conversation["messages"].append({"role": "assistant", "content": np.random.choice(follow_up_responses)})
                
                type_conversations.append(conversation)
            
            # Take only the required number of samples
            conversations.extend(type_conversations[:samples_per_type])
        
        # Save dataset
        output_file = os.path.join(self.output_dir, "conversation_dataset.json")
        with open(output_file, 'w') as f:
            json.dump(conversations, f, indent=2)
        print(f"Conversation dataset saved to {output_file}")
        
        return conversations
    
    def generate_qa_pairs(self, topics, samples_per_topic=50):
        """
        Generate question-answer pairs for chatbot training
        
        Args:
            topics (list): List of topics to generate QA pairs for
            samples_per_topic (int): Number of QA pairs per topic
            
        Returns:
            pandas.DataFrame: DataFrame with questions and answers
        """
        data = {"topic": [], "question": [], "answer": []}
        
        for topic in topics:
            print(f"Generating QA pairs for topic: {topic}")
            
            # Search for relevant content
            search_results = self.web_search(f"{topic} questions and answers", num_results=min(10, samples_per_topic // 5))
            
            topic_qa_pairs = []
            
            for result in search_results:
                scraped_content = self.scrape_webpage(result["url"])
                if "error" not in scraped_content:
                    # Extract existing QA pairs if found
                    if scraped_content.get("qa_pairs"):
                        for qa_pair in scraped_content["qa_pairs"]:
                            topic_qa_pairs.append((qa_pair["question"], qa_pair["answer"]))
                    
                    # Generate QA pairs from paragraphs
                    if scraped_content.get("paragraphs") and len(topic_qa_pairs) < samples_per_topic:
                        for paragraph in scraped_content["paragraphs"]:
                            if len(paragraph) > 100:
                                sentences = paragraph.split('.')
                                if len(sentences) >= 3:
                                    # Create a question from content
                                    sentence = sentences[1].strip()
                                    words = sentence.split()
                                    if len(words) > 5:
                                        # Simple question generation strategies
                                        question_types = ["what", "how", "why", "when", "who"]
                                        q_type = np.random.choice(question_types)
                                        
                                        if q_type == "what":
                                            question = f"What is {' '.join(words[1:3])}?"
                                        elif q_type == "how":
                                            question = f"How does {' '.join(words[1:3])} work?"
                                        elif q_type == "why":
                                            question = f"Why is {' '.join(words[1:3])} important?"
                                        elif q_type == "when":
                                            question = f"When should {' '.join(words[1:3])} be considered?"
                                        else:  # who
                                            question = f"Who benefits from {' '.join(words[1:3])}?"
                                        
                                        topic_qa_pairs.append((question, paragraph))
                
                if len(topic_qa_pairs) >= samples_per_topic:
                    break
            
            # If we don't have enough QA pairs, create synthetic ones
            common_questions = {
                "definition": [
                    f"What is {topic}?",
                    f"How would you define {topic}?",
                    f"Can you explain what {topic} means?"
                ],
                "importance": [
                    f"Why is {topic} important?",
                    f"What's the significance of {topic}?",
                    f"How does {topic} impact us?"
                ],
                "examples": [
                    f"Can you give examples of {topic}?",
                    f"What are some instances of {topic}?",
                    f"What are common {topic} examples?"
                ],
                "comparison": [
                    f"How does {topic} compare to other similar concepts?",
                    f"What makes {topic} different from alternatives?",
                    f"What are the advantages of {topic}?"
                ]
            }
            
            common_answers = {
                "definition": [
                    f"{topic.capitalize()} refers to a concept that involves several key aspects and principles.",
                    f"In simple terms, {topic} is a methodology or approach that focuses on specific elements.",
                    f"{topic.capitalize()} can be defined as a systematic process designed to achieve particular outcomes."
                ],
                "importance": [
                    f"{topic.capitalize()} plays a crucial role in modern contexts due to its significant impact on various areas.",
                    f"The importance of {topic} cannot be overstated, as it influences multiple domains and practices.",
                    f"{topic.capitalize()} is essential because it provides fundamental benefits and advantages."
                ],
                "examples": [
                    f"Common examples of {topic} include various implementations and applications in different contexts.",
                    f"Several notable instances of {topic} can be observed in contemporary settings.",
                    f"Examples of {topic} range from basic applications to more complex implementations."
                ],
                "comparison": [
                    f"Compared to alternatives, {topic} offers distinct advantages such as improved efficiency and effectiveness.",
                    f"{topic.capitalize()} differs from similar concepts in several key ways, including its approach and methodology.",
                    f"The main differences between {topic} and alternatives involve fundamental principles and applications."
                ]
            }
            
            while len(topic_qa_pairs) < samples_per_topic:
                # Generate a synthetic QA pair
                question_type = np.random.choice(list(common_questions.keys()))
                question = np.random.choice(common_questions[question_type])
                answer = np.random.choice(common_answers[question_type])
                
                topic_qa_pairs.append((question, answer))
            
            # Take only the required number of samples
            selected_pairs = topic_qa_pairs[:samples_per_topic]
            
            # Add to dataset
            for question, answer in selected_pairs:
                data["topic"].append(topic)
                data["question"].append(question)
                data["answer"].append(answer)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save dataset
        output_file = os.path.join(self.output_dir, "qa_pairs_dataset.csv")
        df.to_csv(output_file, index=False)
        print(f"QA pairs dataset saved to {output_file}")
        
        return df
    
    # Rest of the class remains the same...
    def generate_intent_dataset(self, intents, samples_per_intent=50):
        """
        Generate an intent classification dataset for chatbots
        
        Args:
            intents (dict): Dictionary mapping intent names to descriptions
            samples_per_intent (int): Number of samples per intent
            
        Returns:
            pandas.DataFrame: DataFrame with utterances and intents
        """
        data = {"utterance": [], "intent": []}
        
        for intent_name, intent_description in intents.items():
            print(f"Generating data for intent: {intent_name}")
            
            # Search for related examples
            search_results = self.web_search(f"{intent_description} examples phrases", num_results=min(5, samples_per_intent // 10))
            
            intent_utterances = []
            
            for result in search_results:
                scraped_content = self.scrape_webpage(result["url"])
                if "error" not in scraped_content and scraped_content.get("paragraphs"):
                    # Look for short sentences that might be examples
                    for paragraph in scraped_content["paragraphs"]:
                        sentences = paragraph.split('.')
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if 5 < len(sentence) < 100 and sentence.endswith('?') and intent_name.lower() in sentence.lower():
                                intent_utterances.append(sentence)
                            elif 5 < len(sentence) < 100 and any(kw in sentence.lower() for kw in intent_description.lower().split()):
                                if np.random.random() < 0.3:  # Only add some sentences to avoid too much noise
                                    intent_utterances.append(sentence)
            
            # Generate variations based on intent
            base_utterances = {
                "greeting": [
                    "Hello there", "Hi", "Hey", "Good morning", "Good afternoon",
                    "Greetings", "Hi there", "Hello"
                ],
                "farewell": [
                    "Goodbye", "Bye", "See you later", "Have a nice day", "Farewell",
                    "Take care", "Until next time", "Bye bye"
                ],
                "thanks": [
                    "Thank you", "Thanks", "I appreciate it", "Thanks a lot",
                    "Thank you so much", "Much appreciated", "Thanks for your help"
                ],
                "help": [
                    "I need help", "Can you help me?", "Help me please", "I'm stuck",
                    "Could you assist me?", "I need assistance", "Help required"
                ],
                "information": [
                    "Tell me about", "I want to know about", "What is", "Information on",
                    "Explain", "Details about", "Can you provide information on"
                ],
                "booking": [
                    "I want to book", "Make a reservation", "Can I book", "Reserve",
                    "I'd like to schedule", "Book an appointment", "Make a booking"
                ],
                "weather": [
                    "What's the weather like", "How's the weather", "Weather forecast",
                    "Is it going to rain", "Temperature today", "Weather report"
                ],
                "time": [
                    "What time is it", "Current time", "Tell me the time",
                    "Do you know what time it is", "Time right now"
                ]
            }
            
            # If we have predefined utterances for this intent
            if intent_name.lower() in base_utterances:
                utterance_templates = base_utterances[intent_name.lower()]
                
                # Generate variations
                for template in utterance_templates:
                    intent_utterances.append(template)
                    
                    # Add locations/objects if appropriate
                    if any(kw in intent_name.lower() for kw in ["booking", "information", "weather"]):
                        objects = ["restaurant", "hotel", "flight", "ticket", "tour", "car", "vacation", "trip"]
                        locations = ["New York", "Paris", "Tokyo", "London", "Sydney", "Berlin", "Rome", "Madrid"]
                        
                        for _ in range(3):  # Generate 3 variations
                            obj = np.random.choice(objects)
                            loc = np.random.choice(locations)
                            
                            if "booking" in intent_name.lower():
                                intent_utterances.append(f"{template} a {obj} in {loc}")
                            elif "information" in intent_name.lower():
                                intent_utterances.append(f"{template} {obj}s in {loc}")
                            elif "weather" in intent_name.lower():
                                intent_utterances.append(f"{template} in {loc}")
            
            # Add variations and modifiers to all intents
            base_utterances = intent_utterances.copy()
            for utterance in base_utterances:
                # Add politeness modifiers
                if np.random.random() < 0.3:
                    polite_prefixes = ["Please ", "Could you ", "Would you mind ", "I'd like to ", "I was wondering if "]
                    intent_utterances.append(f"{np.random.choice(polite_prefixes)}{utterance.lower()}")
                
                # Add urgency modifiers
                if np.random.random() < 0.2:
                    urgent_prefixes = ["Urgently ", "As soon as possible ", "Right now ", "Immediately "]
                    intent_utterances.append(f"{np.random.choice(urgent_prefixes)}{utterance.lower()}")
            
            # Remove duplicates and limit to samples_per_intent
            intent_utterances = list(set(intent_utterances))
            
            # If we still don't have enough, create more synthetic ones
            while len(intent_utterances) < samples_per_intent:
                if intent_name.lower() in base_utterances:
                    template = np.random.choice(base_utterances[intent_name.lower()])
                    modifiers = ["", "Please ", "Can you ", "I'd like to ", "I want to ", "Could you "]
                    intent_utterances.append(f"{np.random.choice(modifiers)}{template.lower()}")
                else:
                    # Generic utterance based on intent name
                    templates = [
                        f"I want to {intent_name.lower()}",
                        f"Can you help me with {intent_name.lower()}",
                        f"I need {intent_name.lower()}",
                        f"How do I {intent_name.lower()}",
                        f"Please provide {intent_name.lower()}"
                    ]
                    intent_utterances.append(np.random.choice(templates))
            
            # Take only the required number of samples
            selected_utterances = intent_utterances[:samples_per_intent]
            
            # Add to dataset
            for utterance in selected_utterances:
                data["utterance"].append(utterance)
                data["intent"].append(intent_name)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save dataset
        output_file = os.path.join(self.output_dir, "intent_dataset.csv")
        df.to_csv(output_file, index=False)
        print(f"Intent dataset saved to {output_file}")
        
        return df
    
    def generate_entity_dataset(self, entities, samples=100):
        """
        Generate an entity extraction dataset for chatbots
        
        Args:
            entities (dict): Dictionary mapping entity types to examples
            samples (int): Total number of samples to generate
            
        Returns:
            list: List of dictionaries with text and entities
        """
        dataset = []
        
        # Generate templates for different entity combinations
        templates = [
            "I want to {action} {object} in {location} on {date}",
            "Can you help me {action} {object} at {location}?",
            "I need a {object} {action} for {date} in {location}",
            "Is there a way to {action} {object} before {date}?",
            "Looking for a {object} to {action} near {location}",
            "When can I {action} {object} in {location}?",
            "Do you know if I can {action} {object} on {date}?",
            "Please {action} this {object} for me at {location}",
            "I'd like to {action} a {object} for {date}",
            "Can you recommend a {object} to {action} in {location}?"
        ]
        
        samples_per_template = samples // len(templates) + 1
        
        for template in templates:
            for _ in range(samples_per_template):
                # Fill in template with random entities
                text = template
                text_entities = []
                
                # Identify entity placeholders in template
                placeholder_entities = {}
                for entity_type, examples in entities.items():
                    placeholder = f"{{{entity_type}}}"
                    if placeholder in template:
                        # Select a random example for this entity type
                        entity_value = np.random.choice(examples)
                        placeholder_entities[placeholder] = (entity_type, entity_value)
                
                # Replace placeholders with actual values and track positions
                for placeholder, (entity_type, entity_value) in placeholder_entities.items():
                    if placeholder in text:
                        start_idx = text.find(placeholder)
                        # Replace placeholder with entity value
                        text = text.replace(placeholder, entity_value, 1)
                        end_idx = start_idx + len(entity_value)
                        # Record entity position
                        text_entities.append({
                            "start": start_idx,
                            "end": end_idx,
                            "type": entity_type,
                            "value": entity_value
                        })
                
                # Add to dataset
                if text_entities:  # Only add if we have entities
                    dataset.append({
                        "text": text,
                        "entities": text_entities
                    })
                
                if len(dataset) >= samples:
                    break
            
            if len(dataset) >= samples:
                break
        
        # Save dataset
        output_file = os.path.join(self.output_dir, "entity_dataset.json")
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Entity extraction dataset saved to {output_file}")
        
        return dataset
    
    def generate_dialog_flow_dataset(self, dialog_flows, samples_per_flow=20):
        """
        Generate a multi-turn dialog dataset for chatbot training
        
        Args:
            dialog_flows (dict): Dictionary mapping flow names to flow descriptions
            samples_per_flow (int): Number of dialog samples per flow
            
        Returns:
            list: List of dialog flow examples
        """
        dialogs = []
        
        for flow_name, flow_description in dialog_flows.items():
            print(f"Generating dialogs for flow: {flow_name}")
            
            # Define templates for different dialog flows
            templates = {
                "booking": [
                    [
                        {"role": "user", "content": "I want to make a booking"},
                        {"role": "assistant", "content": "I'd be happy to help you make a booking. What type of booking are you interested in?"},
                        {"role": "user", "content": "I want to book a {service}"},
                        {"role": "assistant", "content": "Great choice! When would you like to book the {service}?"},
                        {"role": "user", "content": "On {date}"},
                        {"role": "assistant", "content": "Perfect. How many people will be {activity}?"},
                        {"role": "user", "content": "{number} people"},
                        {"role": "assistant", "content": "Thank you. I've booked a {service} for {number} people on {date}. Is there anything else you need?"},
                        {"role": "user", "content": "No, that's all. Thank you!"},
                        {"role": "assistant", "content": "You're welcome! Your booking is confirmed. Enjoy your {service}!"}
                    ],
                    [
                        {"role": "user", "content": "I need to make a reservation"},
                        {"role": "assistant", "content": "I can help you with that. What would you like to reserve?"},
                        {"role": "user", "content": "A {service}"},
                        {"role": "assistant", "content": "When would you like to reserve the {service}?"},
                        {"role": "user", "content": "{date}"},
                        {"role": "assistant", "content": "And how many people will be attending?"},
                        {"role": "user", "content": "{number}"},
                        {"role": "assistant", "content": "Great. I've reserved a {service} for {number} people on {date}. Anything else?"},
                        {"role": "user", "content": "That's all"},
                        {"role": "assistant", "content": "Perfect! Your reservation is confirmed. Have a great time!"}
                    ]
                ],
                "customer_support": [
                    [
                        {"role": "user", "content": "I have a problem with my {product}"},
                        {"role": "assistant", "content": "I'm sorry to hear that you're having an issue with your {product}. Could you please describe the problem in more detail?"},
                        {"role": "user", "content": "It {issue}"},
                        {"role": "assistant", "content": "Thank you for providing that information. Have you tried {solution} yet?"},
                        {"role": "user", "content": "No, I haven't"},
                        {"role": "assistant", "content": "I recommend trying to {solution}. This often resolves the {issue} issue with {product}s."},
                        {"role": "user", "content": "OK, I'll try that"},
                        {"role": "assistant", "content": "Great! Please let me know if that resolves your issue. If not, we can explore other solutions."},
                        {"role": "user", "content": "It worked! Thank you"},
                        {"role": "assistant", "content": "I'm glad to hear that! If you have any other questions or issues, feel free to ask."}
                    ],
                    [
                        {"role": "user", "content": "My {product} isn't working properly"},
                        {"role": "assistant", "content": "I'm sorry to hear about your {product} issue. What seems to be the problem?"},
                        {"role": "user", "content": "It {issue} when I try to use it"},
                        {"role": "assistant", "content": "That sounds frustrating. One common solution is to {solution}. Have you tried that?"},
                        {"role": "user", "content": "No, I haven't tried that yet"},
                        {"role": "assistant", "content": "Please try to {solution} and let me know if that fixes the issue with your {product}."},
                        {"role": "user", "content": "That fixed it!"},
                        {"role": "assistant", "content": "Excellent! I'm glad we were able to resolve your {product} issue. Is there anything else I can help with today?"},
                        {"role": "user", "content": "No, that's all. Thank you for your help!"},
                        {"role": "assistant", "content": "You're welcome! Feel free to reach out if you have any other questions or concerns in the future."}
                    ]
                ],
                "information": [
                    [
                        {"role": "user", "content": "Can you tell me about {topic}?"},
                        {"role": "assistant", "content": "I'd be happy to tell you about {topic}. {topic} is {description}. Would you like to know more about any specific aspect?"},
                        {"role": "user", "content": "Yes, tell me about {aspect}"},
                        {"role": "assistant", "content": "Regarding {aspect} of {topic}, {aspect_details}. Does that answer your question?"},
                        {"role": "user", "content": "Can you give me some examples?"},
                        {"role": "assistant", "content": "Certainly! Some examples of {topic} include {examples}. Would you like more examples or information?"},
                        {"role": "user", "content": "That's helpful, thanks"},
                        {"role": "assistant", "content": "You're welcome! If you have any other questions about {topic} or anything else, feel free to ask."}
                    ],
                    [
                        {"role": "user", "content": "I want to learn more about {topic}"},
                        {"role": "assistant", "content": "{topic} is {description}. It's an interesting subject with many aspects to explore."},
                        {"role": "user", "content": "What are the main benefits of {topic}?"},
                        {"role": "assistant", "content": "The main benefits of {topic} include {benefits}. These advantages make it valuable in many contexts."},
                        {"role": "user", "content": "Are there any downsides?"},
                        {"role": "assistant", "content": "Yes, some potential challenges or limitations of {topic} include {limitations}. However, many of these can be addressed with proper approaches."},
                        {"role": "user", "content": "Thank you for the information"},
                        {"role": "assistant", "content": "You're welcome! I'm glad I could help you learn more about {topic}. Let me know if you have any other questions."}
                    ]
                ]
            }
            
            # Define slot fillers for the templates
            slot_fillers = {
                "booking": {
                    "service": ["hotel room", "restaurant table", "spa appointment", "meeting room", "car rental", "tour guide", "event ticket", "flight"],
                    "date": ["tomorrow", "next Monday", "January 15th", "this weekend", "next week", "May 20th", "December 3rd", "in two weeks"],
                    "activity": ["staying", "dining", "participating", "attending", "visiting"],
                    "number": ["2", "3", "4", "5", "6", "10", "a family of 4", "a group of 8"]
                },
                "customer_support": {
                    "product": ["smartphone", "laptop", "TV", "washing machine", "smart watch", "headphones", "printer", "camera", "refrigerator"],
                    "issue": ["keeps shutting down", "won't turn on", "makes strange noises", "has connection problems", "shows an error message", "is overheating", "isn't charging", "freezes regularly"],
                    "solution": ["restart the device", "check the power connection", "update the software", "clear the cache", "reset to factory settings", "run the troubleshooting wizard", "check for loose connections", "reinstall the drivers"]
                },
                "information": {
                    "topic": ["artificial intelligence", "renewable energy", "digital marketing", "cybersecurity", "cloud computing", "blockchain technology", "machine learning", "internet of things"],
                    "description": ["a rapidly evolving field with significant impact on various industries", "an innovative approach to solving complex problems", "a transformative technology changing how we interact with information", "a fundamental concept in modern technological development"],
                    "aspect": ["applications", "benefits", "history", "future trends", "ethical considerations", "technical requirements", "implementation challenges"],
                    "aspect_details": ["it involves several specialized techniques and methodologies", "it has revolutionized traditional approaches in significant ways", "it continues to evolve with new research and technological advancements", "it requires specific conditions and resources to be implemented effectively"],
                    "examples": ["smart assistants like Siri and Alexa, recommendation systems on streaming platforms, and automated translation services", "Tesla's self-driving technology, IBM's Watson, and fraud detection systems used by financial institutions", "voice recognition software, predictive text algorithms, and automated customer service chatbots"],
                    "benefits": ["increased efficiency, cost reduction, and improved accuracy", "enhanced user experiences, better decision-making capabilities, and scalability", "time savings, error reduction, and the ability to handle complex tasks automatically"],
                    "limitations": ["implementation costs, technical complexity, and potential security concerns", "the need for specialized expertise, integration challenges with existing systems, and ongoing maintenance requirements", "privacy considerations, the risk of system failures, and potential job displacement in certain sectors"]
                }
            }
            
            flow_dialogs = []
            
           # Use flow-specific templates if available, or default to information templates
        flow_templates = templates.get(flow_name.lower(), templates["information"])
        flow_fillers = slot_fillers.get(flow_name.lower(), slot_fillers["information"])
        
        # PERBAIKAN: Pastikan flow_templates bukan kosong dan berbentuk list
        if not flow_templates or not isinstance(flow_templates, list):
            # Gunakan template default jika tidak ada yang sesuai
            flow_templates = templates["information"]
        
        for _ in range(samples_per_flow):
            # PERBAIKAN: Cek apakah flow_templates tidak kosong sebelum memilih secara acak
            if flow_templates:
                template_index = np.random.randint(0, len(flow_templates))
                template = flow_templates[template_index]
                
                # Create a copy to fill in
                dialog = []
                for turn in template:
                    dialog_turn = turn.copy()
                    
                    # Fill in slots
                    for slot in flow_fillers:
                        placeholder = f"{{{slot}}}"
                        if placeholder in dialog_turn["content"]:
                            value = np.random.choice(flow_fillers[slot])
                            dialog_turn["content"] = dialog_turn["content"].replace(placeholder, value)
                    
                    dialog.append(dialog_turn)
                
                # Add dialog to list
                flow_dialogs.append({
                    "flow_type": flow_name,
                    "description": flow_description,
                    "dialog": dialog
                })
        
        dialogs.extend(flow_dialogs)
        
        # Save dataset
        output_file = os.path.join(self.output_dir, "dialog_flow_dataset.json")
        with open(output_file, 'w') as f:
            json.dump(dialogs, f, indent=2)
        print(f"Dialog flow dataset saved to {output_file}")
        
        return dialogs
    
    def split_datasets(self, test_size=0.2, valid_size=0.1):
        """
        Split generated datasets into train/validation/test sets
        
        Args:
            test_size (float): Proportion of data to use for testing
            valid_size (float): Proportion of data to use for validation
            
        Returns:
            dict: Dictionary with split datasets
        """
        split_data = {}
        
        # Process conversation dataset
        conv_path = os.path.join(self.output_dir, "conversation_dataset.json")
        if os.path.exists(conv_path):
            with open(conv_path, 'r') as f:
                conversations = json.load(f)
            
            # Split conversations
            conv_train, conv_test = train_test_split(conversations, test_size=test_size, random_state=42)
            conv_train, conv_valid = train_test_split(conv_train, test_size=valid_size/(1-test_size), random_state=42)
            
            # Save splits
            for name, data in [("train", conv_train), ("valid", conv_valid), ("test", conv_test)]:
                output = os.path.join(self.output_dir, f"conversation_{name}.json")
                with open(output, 'w') as f:
                    json.dump(data, f, indent=2)
            
            split_data["conversations"] = {
                "train": conv_train,
                "valid": conv_valid,
                "test": conv_test
            }
        
        # Process QA pairs dataset
        qa_path = os.path.join(self.output_dir, "qa_pairs_dataset.csv")
        if os.path.exists(qa_path):
            qa_df = pd.read_csv(qa_path)
            
            # Split QA pairs
            train_df, test_df = train_test_split(qa_df, test_size=test_size, random_state=42)
            train_df, valid_df = train_test_split(train_df, test_size=valid_size/(1-test_size), random_state=42)
            
            # Save splits
            for name, data in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
                output = os.path.join(self.output_dir, f"qa_pairs_{name}.csv")
                data.to_csv(output, index=False)
            
            split_data["qa_pairs"] = {
                "train": train_df,
                "valid": valid_df,
                "test": test_df
            }
        
        # Process intent dataset
        intent_path = os.path.join(self.output_dir, "intent_dataset.csv")
        if os.path.exists(intent_path):
            intent_df = pd.read_csv(intent_path)
            
            # Split by intent to ensure representation of all classes
            train_df, test_df = train_test_split(intent_df, test_size=test_size, stratify=intent_df["intent"], random_state=42)
            train_df, valid_df = train_test_split(train_df, test_size=valid_size/(1-test_size), stratify=train_df["intent"], random_state=42)
            
            # Save splits
            for name, data in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
                output = os.path.join(self.output_dir, f"intent_{name}.csv")
                data.to_csv(output, index=False)
            
            split_data["intents"] = {
                "train": train_df,
                "valid": valid_df,
                "test": test_df
            }
        
        # Process entity dataset
        entity_path = os.path.join(self.output_dir, "entity_dataset.json")
        if os.path.exists(entity_path):
            with open(entity_path, 'r') as f:
                entities = json.load(f)
            
            # Split entities
            entity_train, entity_test = train_test_split(entities, test_size=test_size, random_state=42)
            entity_train, entity_valid = train_test_split(entity_train, test_size=valid_size/(1-test_size), random_state=42)
            
            # Save splits
            for name, data in [("train", entity_train), ("valid", entity_valid), ("test", entity_test)]:
                output = os.path.join(self.output_dir, f"entity_{name}.json")
                with open(output, 'w') as f:
                    json.dump(data, f, indent=2)
            
            split_data["entities"] = {
                "train": entity_train,
                "valid": entity_valid,
                "test": entity_test
            }
        
        # Process dialog flow dataset
        dialog_path = os.path.join(self.output_dir, "dialog_flow_dataset.json")
        if os.path.exists(dialog_path):
            with open(dialog_path, 'r') as f:
                dialogs = json.load(f)
            
            # Split dialogs
            dialog_train, dialog_test = train_test_split(dialogs, test_size=test_size, random_state=42)
            dialog_train, dialog_valid = train_test_split(dialog_train, test_size=valid_size/(1-test_size), random_state=42)
            
            # Save splits
            for name, data in [("train", dialog_train), ("valid", dialog_valid), ("test", dialog_test)]:
                output = os.path.join(self.output_dir, f"dialog_flow_{name}.json")
                with open(output, 'w') as f:
                    json.dump(data, f, indent=2)
            
            split_data["dialogs"] = {
                "train": dialog_train,
                "valid": dialog_valid,
                "test": dialog_test
            }
        
        print(f"All datasets split and saved to {self.output_dir}")
        return split_data
    
    def generate_all_datasets(self, config=None):
        """
        Generate all datasets with default or specified configuration
        
        Args:
            config (dict, optional): Configuration dictionary with dataset parameters
            
        Returns:
            dict: Dictionary with all generated datasets
        """
        if config is None:
            # Default configuration
            config = {
                "conversation_types": ["casual", "technical", "customer_service", "educational"],
                "qa_topics": ["technology", "science", "business", "health", "education"],
                "intents": {
                    "greeting": "user greetings and hello messages",
                    "farewell": "user goodbyes and end conversations",
                    "help": "user requesting assistance or help",
                    "information": "user requesting information about a topic",
                    "booking": "user wanting to make a reservation or booking",
                    "weather": "user asking about weather conditions",
                    "time": "user asking about time or date"
                },
                "entities": {
                    "action": ["book", "reserve", "order", "schedule", "cancel", "modify", "purchase", "rent"],
                    "object": ["room", "table", "car", "flight", "ticket", "appointment", "tour", "service"],
                    "location": ["New York", "London", "Paris", "Tokyo", "Sydney", "Berlin", "Rome", "Madrid"],
                    "date": ["tomorrow", "next week", "January 15", "this weekend", "Monday", "in two days"]
                },
                "dialog_flows": {
                    "booking": "Conversations about making reservations or bookings",
                    "customer_support": "Customer service and troubleshooting conversations",
                    "information": "Conversations requesting and providing information"
                },
                "samples": {
                    "conversations_per_type": 20,
                    "qa_pairs_per_topic": 30,
                    "samples_per_intent": 50,
                    "entity_samples": 200,
                    "dialogs_per_flow": 15
                }
            }
        
        print("Starting dataset generation process...")
        results = {}
        
        # Generate conversation dataset
        print("\n--- Generating conversation dataset ---")
        conversations = self.generate_conversation_dataset(
            config["conversation_types"], 
            samples_per_type=config["samples"]["conversations_per_type"]
        )
        results["conversations"] = conversations
        
        # Generate QA pairs dataset
        print("\n--- Generating QA pairs dataset ---")
        qa_pairs = self.generate_qa_pairs(
            config["qa_topics"], 
            samples_per_topic=config["samples"]["qa_pairs_per_topic"]
        )
        results["qa_pairs"] = qa_pairs
        
        # Generate intent classification dataset
        print("\n--- Generating intent classification dataset ---")
        intents = self.generate_intent_dataset(
            config["intents"], 
            samples_per_intent=config["samples"]["samples_per_intent"]
        )
        results["intents"] = intents
        
        # Generate entity extraction dataset
        print("\n--- Generating entity extraction dataset ---")
        entities = self.generate_entity_dataset(
            config["entities"], 
            samples=config["samples"]["entity_samples"]
        )
        results["entities"] = entities
        
        # Generate dialog flow dataset
        print("\n--- Generating dialog flow dataset ---")
        dialogs = self.generate_dialog_flow_dataset(
            config["dialog_flows"], 
            samples_per_flow=config["samples"]["dialogs_per_flow"]
        )
        results["dialogs"] = dialogs
        
        # Split datasets
        print("\n--- Splitting datasets into train/validation/test sets ---")
        split_data = self.split_datasets()
        results["splits"] = split_data
        
        print("\nAll datasets generated successfully!")
        return results


# Example usage
if __name__ == "__main__":
    generator = ChatAIDatasetGenerator(output_dir="chat_datasets")
    
    # Generate all datasets with default configuration
    datasets = generator.generate_all_datasets()
    
    # Or generate specific datasets
    # Custom configuration
    custom_config = {
        "conversation_types": ["technical", "customer_service"],
        "qa_topics": ["artificial intelligence", "machine learning"],
        "intents": {
            "greeting": "user greetings",
            "farewell": "user goodbyes",
            "help": "user requesting assistance"
        },
        "entities": {
            "action": ["analyze", "process", "train", "deploy"],
            "object": ["model", "algorithm", "dataset", "neural network"],
            "location": ["cloud", "local", "server", "edge device"],
            "date": ["today", "this week", "next month"]
        },
        "dialog_flows": {
            "technical_support": "AI system troubleshooting conversations",
            "training": "Conversations about training AI models"
        },
        "samples": {
            "conversations_per_type": 10,
            "qa_pairs_per_topic": 20,
            "samples_per_intent": 30,
            "entity_samples": 100,
            "dialogs_per_flow": 10
        }
    }
    
    # Generate with custom configuration
    # custom_datasets = generator.generate_all_datasets(config=custom_config)
    
    # Or generate individual datasets
    # conversations = generator.generate_conversation_dataset(["casual", "technical"], samples_per_type=15)
    # qa_pairs = generator.generate_qa_pairs(["technology", "science"], samples_per_topic=25)