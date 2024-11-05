import requests
from bs4 import BeautifulSoup
from config import all_config
import json
import os

bad_courses = []

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
def get_semesters(course_id):
    base_url = "https://www3.openu.ac.il/ouweb/owal/catalog.sel_list_semesters?kurs_in="
    slink = base_url + course_id
    response = requests.get(slink)
    if response.status_code != 200:
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    try:
        content = soup.find('ul', {'class': 'list_square'})
        if not content:
            return None
        
        semesters = []
        for li in content.find_all('li'):
            semesters.append(clean_text(li.get_text()))
        return semesters

    except Exception as e:
        # Print exception
        print(f"Error parsing course: {slink}")
        print(e)     
        return None


##############################################################################
def get_overlap_courses(course_id, url):
    dup_url = f"https://www3.openu.ac.il/ouweb/owal/chofef.list?kurs_p=00000&machlaka_akademit_p=000&daf_p=0&sug_p=I&p_search={course_id}"
    response = requests.get(dup_url)
    if response.status_code != 200:
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    duplicates = []
    try:
        items = soup.find_all('td', {'width': '13%'})
        if not items:
            return None
        
        for item in items:
            duplicates.append(item.get_text().strip())

        # remove course_id from the list
        duplicates.remove(course_id)
        return duplicates

    except Exception as e:
        # Print exception
        print(f"Error parsing course duplicates: {dup_url}")
        print(e)     
        return None


##############################################################################
# The course URL should be formatted like: https://www.openu.ac.il/courses/12345.htm
# extract the course number
def extract_course_number_from_course_url(course_url):
    return course_url.split('/')[-1].split('.')[0]

def extract_course_number_from_dependency_line(element):
    course_numbers = []
    condition = element.find_all('a')
    for course in condition:
        # if the course element contains an 'href' attribute
        if course.get('href'):
            course_number = extract_course_number_from_course_url(course['href'])
            course_numbers.append(course_number)
    return course_numbers


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
            "classification": [],
            "condition_text": "",
            "condition_courses": [],
            "required_deps_text": "",
            "required_deps_courses": [],
            "recommended_deps_text": "",
            "recommended_deps_courses": [],
            "overlap_url": "",
            "overlap_courses": [],
            "text": [],
            "footnotes": []
        }

        course_data['course_url'] = course_url
        
        # Is the course discontinued?
        course_data['discontinued'] = False
        title2 = soup.find('div', {'class': 'titles2'})
        if (title2):
            t2text = title2.get_text().strip()
            if t2text == "הקורס אינו מוצע עוד":
                course_data['discontinued'] = True
                course_data['text'].append(t2text)

        content = soup.find('div', {'class': 'content'})
        if (not content):
            content = soup.find('div', {'id': 'content'})
        if ( not content):
            print('No content found')
            return None
        
        classifications = []
        course_id2 = None
        course_name2 = None
        # Loop over the children of the content tag.
        for child in content.children:

            if (child == '\n'):
                continue

            # get the text of the child elewment
            child_text = child.get_text().lstrip()

            # Course text includes everything in content.
            course_data['text'].append(clean_text(child_text))

            # if the child tag is 'h1' and the id is 'course_title'
            if child.name == 'h1' and child.get('id') == 'course_title':
                title_text = clean_text(child_text)
                course_id2, course_name2 = title_text.split(' ', 1)
#                # Ignore. We use the class='koterettitle' instead.
                continue
            elif child.name == 'p':
                child_class = child.get('class')
                if (child_class and 'koterettitle' in child_class):
                    title_text = clean_text(child_text)
                    if title_text == "הקורס אינו מוצע עוד":
                        course_data['discontinued'] = True
                        course_data['text'].append(title_text)
                    else:
                        course_data['course_id'], course_data['course_name'] = title_text.split(' ', 1)
                    continue
                elif child_text.startswith('נקודות זכות'):
                    course_data['credits'] = clean_text(child_text)
                    continue
                elif child_text.startswith('שיוך:'):
                    classifications.append(clean_text(child_text.replace('שיוך:', '')))
                    continue
                elif child_text.startswith('שיוך נוסף:'):
                    classifications.append(clean_text(child_text.replace('שיוך נוסף:', '')))
                    continue
                elif child_text.startswith('ידע קודם מומלץ'):
                    course_data['recommended_deps_text'] = clean_text(child_text.replace('ידע קודם מומלץ:', ''))
                    course_data['recommended_deps_courses'] = extract_course_number_from_dependency_line(child)
                    continue
                elif child_text.startswith('ידע קודם דרוש'):
                    course_data['required_deps_text'] = clean_text(child_text.replace('ידע קודם דרוש:', ''))
                    course_data['required_deps_courses'] = extract_course_number_from_dependency_line(child)
                    continue
                elif child_text.startswith('תנאי קבלה'):
                    if course_data['condition_text']:
                        # Ignore if condition text already exists - it will be in the full text
                        continue
                    course_data['condition_text'] = clean_text(child_text.replace('תנאי קבלה:', ''))
                    course_data['condition_courses'] = extract_course_number_from_dependency_line(child)
                    continue
                elif child_class and ('textheara' in child_class or 'footnotetext' in child_class):
                    a_tag = child.find('a', {'id': 'hofefim'})
                    if a_tag:
                        course_data['overlap_url'] = a_tag['href'] 
                    else:
                        course_data['footnotes'].append(clean_text(child_text))
                    continue

        if course_id2 and not course_data['course_id']:
            course_data['course_id'] = course_id2
        if course_name2 and not course_data['course_name']:
            course_data['course_name'] = course_name2
        # check if course id is a number
        if not course_data['course_id'].isdigit():
            # get course number from the url
            course_data['course_name'] = course_data['course_id'] + ' ' + course_data['course_name']
            course_data['course_id'] = course_url.split('/')[-1].split('.')[0]

        # Parse classifications
        if classifications:
            for classification in classifications:
                # split classification text by " / "
                hierarchy = []
                for part in classification.split(' / '):
                    hierarchy.append(part)                
                course_data['classification'].append(hierarchy)

        sems = get_semesters(course_data['course_id'])
        if (sems):
            course_data['semesters'] = sems

        if 'overlap_url' in course_data:
            overlap_url = course_data['overlap_url']
            if overlap_url:
                course_data['overlap_courses'] = get_overlap_courses(course_data['course_id'], overlap_url)

        return course_data
    
    except Exception as e:
        # Print exception
        msg = f"Error parsing course: {course_url}: {e}"
        print(msg)
        bad_courses.append(msg)
        return None

##############################################################################
# Main function to extract course data for the range and save to JSON
##############################################################################
def extract_all_course_data(from_id, to_id, is_test=False):
    print('Extracting course data...')
    base_url = "http://www.openu.ac.il/courses/{}.htm"
    courses_data = []
    course_names = {}

    for course_number in range(from_id, to_id):
        course_url = base_url.format(course_number)
        course_data = extract_course_data(course_url)
        if course_data:
            course_names[course_data['course_name']] = course_data['course_id']
            print(f"Course Number: {course_number}, found")
            courses_data.append(course_data)
        else:
            print(f"Course Number: {course_number}")
    
    if not is_test:
        filename = os.path.join(all_config['General']['DB_Path'], 'all_courses.json')
    else:
        filename = os.path.join(all_config['General']['DB_Path'], 'all_courses_test.json')

    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(courses_data, json_file, ensure_ascii=False, indent=4)

    print("Bad courses:")
    for msg in bad_courses:
        print(msg)





if __name__ == "__main__":
#    extract_all_course_data(10582, 20200, True)
    extract_all_course_data(10100, 34000)
