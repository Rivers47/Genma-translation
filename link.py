import re
import sys

def find_tags_in_file(filename):
    # Compile the regex pattern:
    #  - lec\d+   : 'lec' followed by one or more digits
    #  - :        : a literal colon
    #  - pai\d+   : 'pai' followed by one or more digits
    #  - (?:-\d+)? : optional group of '-' followed by digits
    pattern = re.compile(r"label\{lec\d+:pai(\d+)(?:-(\d+))?\}")
    
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Find all matches in the file text
    matches = pattern.findall(text)

    
    lec_str = "lec6"
    parsed_tags = []
    for tag in matches:
        print(tag)
        start_str, end_str = tag
        if end_str == '':   # no end_str captured => single number
            num = int(start_str)
            original_tag = f"{lec_str}:pai{start_str}"
            parsed_tags.append((num, original_tag))
        else:
            start_num = int(start_str)
            end_num = int(end_str)
            original_tag = f"{lec_str}:pai{start_str}-{end_str}"
            for num in range(start_num, end_num+1):
                parsed_tags.append((num, original_tag))
            
    f.close()
    return parsed_tags


if __name__ == "__main__":
    # Replace 'input.txt' with the path to your file
    
    filename = 'lecture6.tex' #sys.argv[1]
    found_tags = find_tags_in_file(filename)
    pattern = re.compile("[^\{](牌(\d+)(?:-\d+)?)")

    with open(filename, 'r', encoding='utf-8') as f_in:
        content = f_in.read()
    matches = pattern.findall(content)
    replaced_content = content
    for match in matches:
        replaced, idx = match
        replacement = f'\\\\hyperref[{found_tags[int(idx)-1][1]}]{{{replaced}}}'
        replaced_content = re.sub('[^{]'+replaced, replacement, replaced_content)

    with open(f'{filename}.new', 'w', encoding='utf-8') as f_out:
        f_out.write(replaced_content)

    # if found_tags:
    #     print("Found tags:")
    #     for tag in found_tags:
    #         print(tag)
    # else:
    #     print("No matching tags were found.")



