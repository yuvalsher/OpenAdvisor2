import requests
from bs4 import BeautifulSoup
import config
import json

# Function to extract course data from a course page
##############################################################################
def extract_course_data1(course_url):
    response = requests.get(course_url)
    if response.status_code != 200:
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    try:
        course_id_tag = soup.find('h1', {'id': 'course_title'})
        if not course_id_tag:
            return None

        course_id = course_id_tag.text.split()[0]
        course_name = course_id_tag.text[len(course_id):].strip().replace('\u200f', '')  # Remove the course ID and RLM character

        credits_tag = soup.find('p', string=lambda t: t and 'נקודות זכות' in t)
        if not credits_tag:
            return None
        
        try:
            credits = int(credits_tag.text.split()[0])
        except ValueError:
            return None

        level_tag = soup.find('p', string=lambda t: t and 'רמה' in t)
        if not level_tag:
            return None
        
        level = level_tag.text.split()[-1]
        
        dependencies = []
        for prereq in soup.find_all('a', href=True):
            if 'http://www.openu.ac.il/courses/' in prereq['href']:
                dependencies.append(prereq['href'].split('/')[-1].split('.')[0])
        
        return {
            'course_id': course_id,
            'course_name': course_name,
            'credits': credits,
            'level': level,
            'dependencies': dependencies
        }
    except Exception:
        return None


##############################################################################
def clean_text(text):
        return text.replace('\u200e', '').replace('\u200f', '').strip()

##############################################################################
def extract_course_data(course_url):
    response = requests.get(course_url)
    if response.status_code != 200:
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    try:
    
        course_data = {
            "course_url": "",
            "course_id": "",
            "course_name": "",
            "credits": "",
            "primary_classification": "",
            "secondary_classification": [],
            "required_dependencies": [],
            "recommended_dependencies": [],
            "overlap": "",
            "text": [],
            "footnotes": []
        }

        course_data['course_url'] = course_url
        
        content = soup.find('div', {'class': 'content'})
        if (not content):
            content = soup.find('div', {'id': 'content'})
        if ( not content):
            print('No content found')
            return None
        
        # Loop over the children of the content tag.
        for child in content.children:

            if (child == '\n'):
                continue

            # get the text of the child elewment
            child_text = child.get_text().lstrip()

            # if the child tag is 'h1' and the id is 'course_title'
            if child.name == 'h1' and child.get('id') == 'course_title':
                # Ignore. We use the class='koterettitle' instead.
                continue
            elif child.name == 'p':
                child_class = child.get('class')
                if (child_class and 'koterettitle' in child_class):
                    title_text = clean_text(child_text)
                    course_data['course_id'], course_data['course_name'] = title_text.split(' ', 1)
                    continue
                elif 'נקודות זכות' in child_text:
                    course_data['credits'] = clean_text(child_text)
                    continue
                elif child_text.startswith('שיוך:'):
                    course_data['primary_classification'] = clean_text(child_text.replace('שיוך:', ''))
                    continue
                elif child_text.startswith('שיוך נוסף:'):
                    course_data['secondary_classification'].append(clean_text(child_text.replace('שיוך נוסף:', '')))
                    continue
                elif child_text.startswith('ידע קודם מומלץ'):
                    recommended_courses = child.find_all('a')
                    for course in recommended_courses:
                        course_data['recommended_dependencies'].append(clean_text(course.get_text()))
                    continue
                elif child_text.startswith('ידע קודם דרוש'):
                    required_dependencies = child.find_all('a')
                    for course in required_dependencies:
                        course_data['required_dependencies'].append(clean_text(course.get_text()))
                    continue
                elif child_class and ('textheara' in child_class or 'footnotetext' in child_class):
                    a_tag = child.find('a', {'id': 'hofefim'})
                    if a_tag:
                        course_data['overlap'] = a_tag['href'] 
                    else:
                        course_data['footnotes'].append(clean_text(child_text))
                    continue

            # now handle remaining text elements.
            course_data['text'].append(clean_text(child.get_text()))

        return course_data
    
    except Exception as e:
        # Print exception
        print(f"Error parsing course: {course_url}")
        print(e)     
        return None

##############################################################################
# Main function to extract course data for the range and save to JSON
##############################################################################
def extract_all_course_data():
    print('Extracting course data...')
    base_url = "http://www.openu.ac.il/courses/{}.htm"
    courses_data = []

    for course_number in range(20100, 21000):
        course_url = base_url.format(course_number)
        course_data = extract_course_data(course_url)
        if course_data:
            print(f"Course Number: {course_number}, found")
            courses_data.append(course_data)
        else:
            print(f"Course Number: {course_number}")
    
    with open(config.UPLOAD_FILES_FOLDER + '\\cs_courses.json', 'w', encoding='utf-8') as json_file:
        json.dump(courses_data, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    extract_course_data()
