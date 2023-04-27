#!/usr/bin/env python
# coding: utf-8


class AadhaarExtractor:  # assume inputs are file name and not cv2 images             #give underscore for private member functions
    def __init__(self, data1=None, data2=None):  # and pil images

        self.aadhaar_no = None  # 8 fields now , possibly more later
        self.gender = None
        self.dob = None
        self.address = None
        self.name = None
        self.enrollment_no = None
        self.vid = None
        self.phonenumber = None

        self.maindict = {}  # is variables really needed?
        self.maindict['aadhaar_no'] = self.aadhaar_no
        self.maindict['gender'] = self.gender
        self.maindict['dob'] = self.dob
        self.maindict['address'] = self.address
        self.maindict['name'] = self.name
        self.maindict['enrollment_no'] = self.enrollment_no
        self.maindict['vid'] = self.vid
        self.maindict['phonenumber'] = self.phonenumber

        self.data1 = data1
        self.data2 = data2

        self.extract_details()

    def load(self, data1, data2=None):  # can load and jsonify be merged?
        self.data1 = data1
        self.data2 = data2
        self.extract_details()

    def to_json(self):
        import json
        # try:
        #     f = open(jsonpath, 'r+')
        # except FileNotFoundError:  # for the first time
        #     f = open(jsonpath, 'w+')
        # try:
        #     maindict = json.load(f)
        # except ValueError:  # for the first time
        #     print("value error")
        #     maindict = {}

        # if self.maindict['aadhaar_no'] in maindict.keys():
        #     choice = input(
        #         "This aadhar number is already present in the database:\n Do you want to update the the data for this aadhaar number (y\n)?")
        #     # TODO where will the message be displayed
        #     if choice.lower() == 'n':
        #         f.close()
        #         return self.maindict

        # maindict[self.maindict['aadhaar_no']] = self.maindict
        # f.seek(0)
        # json.dump(maindict, f, indent=2)
        # f.close()
        return self.maindict

    def file_type(self, file):
        # import re
        # if re.match(".*\.pdf$", filePath, re.M | re.I):
        if file.content_type == r'application/pdf':
            return 'pdf'
        # if re.match(".*\.(png|jpg|jpeg|bmp|svg)$", filePath, re.M | re.I):  # changed and made more flexible
        if file.content_type == r'image/jpeg':
            return 'img'
        return 0

    def give_details_back(self, data):

        # if self.file_type(data) == 'pdf':
        #     dict = self.extract_from_pdf(data)
        # elif self.file_type(data) == 'img':
        dict = self.extract_from_images(data)
        # else:
        #     pass
        return dict

    def extract_details(self):
        if self.data1 != None:
            dict1 = self.give_details_back(self.data1)
            for key in dict1.keys():  # use lambdas?
                self.maindict[key] = dict1[key]
        if self.data2 != None:
            dict2 = self.give_details_back(self.data2)
            for key in dict2.keys():
                self.maindict[key] = dict2[key]  # need to check for conflicts?

    def extract_from_images(self, file):
        import numpy as np
        from ultralytics import YOLO
        import cv2
        import pytesseract
        import logging

        logging.basicConfig(level=logging.NOTSET)
        MODEL_PATH = r"C:\Users\91886\Desktop\intership\text_extraction\python files\best.pt"

        def filter_tuples(lst):
            d = {}
            for tup in lst:
                key = tup[1]
                value = tup[2]
                if key not in d:
                    d[key] = (tup, value)
                else:
                    if value > d[key][1]:
                        d[key] = (tup, value)
            return [tup for key, (tup, value) in d.items()]

        def clean_words(name):
            name = name.replace("8", "B")
            name = name.replace("0", "D")
            name = name.replace("6", "G")
            name = name.replace("1", "I")
            name = name.replace('5', 'S')

            return name

        def clean_dob(dob):
            dob = dob.strip()
            dob = dob.replace('l', '/')
            dob = dob.replace('L', '/')
            dob = dob.replace('I', '/')
            dob = dob.replace('i', '/')
            dob = dob.replace('|', '/')
            dob = dob.replace('\"', '/1')
            #       dob = dob.replace(":","")
            dob = dob.replace(" ", "")
            return dob

        def validate_aadhaar_numbers(candidate):
            if candidate == None :
                return True
            candidate = candidate.replace(' ', '')
            # The multiplication table
            d = [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
                [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
                [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
                [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
                [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
                [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
                [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
                [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
                [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
            ]
            # permutation table p
            p = [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
                [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
                [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
                [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
                [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
                [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
                [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
            ]
            # inverse table inv
            inv = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]
            # print("sonddffsddsdd")
            # print(len(candidate))
            lastDigit = candidate[-1]
            c = 0
            array = [int(i) for i in candidate if i != ' ']
            array.pop()
            array.reverse()
            for i in range(len(array)):
                c = d[c][p[((i + 1) % 8)][array[i]]]  # use verheoffs algorithm to validate
            if inv[c] == int(lastDigit):
                return True
            return False

        file.seek(0)
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (640, 640))
        # cv2.imshow("sone", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        model = YOLO(r"best.pt")
        results = model(img)
        # rois = []
        roidata = []
        for result in results:
            #     cls = result.boxes.cls
            #     boxes = result.boxes.xyxy
            boxes = result.boxes
            for box in boxes:
                #     x1,y1,x2,y2 = box
                l = box.boxes.flatten().tolist()
                x1, y1, x2, y2 = list(map(int, l[0:4]))
                confidence, cls = l[4:]
                cls = int(cls)
                #        l = list(box)
                #        x1,y1,x2,y2 = list(map(int,l))
                # print(x1, x2, y1, y2)
                # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(img, str(cls), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                roi = img[y1:y2, x1:x2]
                #        rois.append(roi)
                tple = (roi, cls,confidence)
                roidata.append(tple)
        # cv2.imshow("s", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(roidata)
        index = {0: "aadhaar_no",
                 1: "dob",
                 2: "gender",
                 3: "name",
                 4: "address"}
        logging.info('BEFORE FILTERING :')
        logging.info(len(roidata))
        roidata = filter_tuples(roidata)
        maindict = {}
        maindict['aadhaar_no'] = maindict['dob'] = maindict['gender'] = maindict['address'] = maindict['name'] = maindict['phonenumber'] = maindict['vid'] = maindict['enrollment_number'] = None
        logging.info('AFTER FILTERING :')
        logging.info(len(roidata))
        for tple in roidata:
            cls = tple[1]
            data = tple[0]
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            info = pytesseract.image_to_string(data).strip()
            logging.info(str(cls)+'-'+info)
            # if there are more than one of the same class, then assign it to none :FIXED
            # FIXED: if there more than one of the same class, then select the one with highest confidence
            if cls == 0:
                    maindict['aadhaar_no'] = info

            elif cls == 1:
                    maindict['dob'] = info

            elif cls == 2:
                    maindict['gender'] = info

            elif cls == 3:
                    maindict['name'] = info

            elif cls ==4:
                    maindict['address'] = info

            # extracted text cleaned up :FIXED
        if (maindict['name']!=None):
            maindict['name'] = clean_words(maindict['name'])

        if maindict['dob']!=None:
            maindict['dob'] = clean_dob(maindict['dob'])

        if maindict['aadhaar_no']!=None:
            maindict['aadhaar_no'] = maindict['aadhaar_no'].replace(' ','')
        #  validated aadhaar card number :FIXED
        if maindict['aadhaar_no'] != None:
            if len(maindict['aadhaar_no']) == 12:
                try:
                    if not validate_aadhaar_numbers(maindict['aadhaar_no']) :
                        maindict['aadhar_no'] = "INVALID AADHAAR NUMBER"
                except ValueError:
                    maindict['aadhaar_no'] = None
            else:
                maindict['aadhaar_no'] = None


        # TODO extract these fields too
        # maindict['phonenumber'] = None
        # maindict['vid'] = None
        # maindict['enrollment_no'] = None

        logging.info(maindict)
        return maindict

    def extract_from_pdf(self, file):
        def extract_pymupdf(file):
            # Usinf pymupdf
            import fitz  # this is pymupdf
            # extract text page by page
            with fitz.open(stream=file.stream.read(), filetype='pdf') as doc:
                pymupdf_text = ""
                if (doc.is_encrypted):
                    passw = input("Enter the password")
                    # TODO display this message where?
                    doc.authenticate(password=passw)
                for page in doc:
                    pymupdf_text += page.get_text("Text")
                return pymupdf_text

        def get_details(txt):
            import re

            pattern = re.compile(
                r'Enrolment No\.: (?P<enrolment_no>[^\n]*)\nTo\n[^\n]*\n(?P<name>[^\n]*)\n(?P<relation>[S,W,D])\/O: (?P<fathers_name>[^\n]*)\n(?P<address>.*)(?P<phonenumber>\d{10})\n(?P<aadhaar_number>^\d{4} \d{4} \d{4}\n).*(?P<vid>\d{4} \d{4} \d{4} \d{4})\n.*DOB: (?P<dob>[^\n]*)\n.*(?P<gender>MALE|FEMALE|Female|Male)',
                re.M | re.A | re.S)
            # gets all info in one match(enrolment to V) which can then be parsed by the groups
            return pattern.search(txt)

        def get_enrolment_no(txt):
            return get_details(txt).group('enrolment_no')

        def get_name(txt):
            return get_details(txt).group('name')

        def get_fathers_name(txt):
            matchobj = get_details(txt)
            relation = matchobj.group('fathers_name')
            if matchobj.group('relation').lower() == 'w':
                return None
            return relation

        def get_husbands_name(txt):
            matchobj = get_details(txt)
            return matchobj.group('fathers_name')

        def get_address(txt):
            return get_details(txt).group('address')

        def get_phonenumber(txt):
            return get_details(txt).group('phonenumber')

        def get_aadhaarnumber(txt):
            return get_details(txt).group('aadhaar_number').strip()

        def get_vid(txt):
            return get_details(txt).group('vid')

        def get_gender(txt):
            return get_details(txt).group('gender')

        def get_dob(txt):
            return get_details(txt).group('dob')

        def get_details_pdf(file):
            import re
            txt = extract_pymupdf(file)
            dict = {'vid': get_vid(txt),
                    'enrollment_no': get_enrolment_no(txt),  # Fathers name':get_fathers_name(txt),
                    'name': get_name(txt),  # if dict['Fathers name'] == None :
                    'address': get_address(txt),
                    'phonenumber': get_phonenumber(txt),  # dict['Husbands name']=get_husbands_name(txt)
                    'aadhaar_no': get_aadhaarnumber(txt),
                    'sex': get_gender(txt),
                    'dob': get_dob(txt)}
            #                       ,'ID Type':'Aadhaar'}
            return dict

        return get_details_pdf(file)


if __name__ == '__main__':
    obj = AadhaarExtractor()
    file = open(r"C:\Users\91886\Desktop\cardF.jpg",'r')
    obj.load(file)
    print(obj.to_json())

    # def extract_from_images(self, path):
    #     import json
    #     import pytesseract
    #     import cv2
    #     import numpy as np
    #     import sys
    #     import re
    #     import os
    #     from PIL import Image, ImageOps
    #     import ftfy
    #     import io

    # def detect_angle_front(image):
    #     mask = np.zeros(image.shape, dtype=np.uint8)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     blur = cv2.GaussianBlur(gray, (3, 3), 0)
    #     adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    #
    #     cnts = cv2.findContours(adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #
    #     for c in cnts:
    #         area = cv2.contourArea(c)
    #         if area < 45000 and area > 20:
    #             cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
    #     #     cv2.imwrite("IAMHERE.png",mask)
    #     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #     # cv2_imshow(mask)
    #     h, w = mask.shape
    #
    #     # Horizontal
    #     if w > h:
    #         left = mask[0:h, 0:0 + w // 4]
    #         right = mask[0:h, 3 * w // 4:]
    #         left_pixels = cv2.countNonZero(left)
    #         right_pixels = cv2.countNonZero(right)
    #         return 0 if left_pixels <= right_pixels else 180
    #     # Vertical
    #     else:
    #         top = mask[0:0 + h // 4, 0:w]
    #         bottom = mask[3 * h // 4:, 0:w]
    #         top_pixel = cv2.countNonZero(top)
    #         bottom_pixel = cv2.countNonZero(bottom)
    #         return 90 if top_pixel <= bottom_pixel else 270
    #
    # def detect_angle_back(image):
    #     mask = np.zeros(image.shape, dtype=np.uint8)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     blur = cv2.GaussianBlur(gray, (3, 3), 0)
    #     adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    #
    #     cnts = cv2.findContours(adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #
    #     for c in cnts:
    #         area = cv2.contourArea(c)
    #         if area < 45000 and area > 20:
    #             cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
    #
    #     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #     # cv2_imshow(mask)
    #     h, w = mask.shape
    #
    #     # Horizontal
    #     if w > h:
    #         left = mask[0:h, 0:0 + w // 2]
    #         right = mask[0:h, w // 2:]
    #         left_pixels = cv2.countNonZero(left)
    #         right_pixels = cv2.countNonZero(right)
    #         # print(left_pixels)
    #         # print(right_pixels)
    #         return 0 if left_pixels <= right_pixels else 180
    #     # Vertical
    #     else:
    #         top = mask[0:0 + h // 2, 0:w]
    #         bottom = mask[h // 2:, 0:w]
    #         top_pixels = cv2.countNonZero(top)
    #         bottom_pixels = cv2.countNonZero(bottom)
    #         return 90 if top_pixels <= bottom_pixels else 270
    #
    # def validate_aadhaar_numbers(candidates):
    #     # The multiplication table
    #     d = [
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #         [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    #         [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    #         [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    #         [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    #         [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    #         [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    #         [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    #         [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    #         [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    #     ]
    #     # permutation table p
    #     p = [
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #         [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    #         [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    #         [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    #         [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    #         [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    #         [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    #         [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
    #     ]
    #     # inverse table inv
    #     inv = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]
    #
    #     for candidate in candidates:
    #         lastDigit = candidate[-1]
    #         c = 0
    #         array = [int(i) for i in candidate if i != ' ']
    #         array.pop()
    #         array.reverse()
    #         for i in range(len(array)):
    #             c = d[c][p[((i + 1) % 8)][array[i]]]  # use verheoffs algorithm to validate
    #         if inv[c] == int(lastDigit):
    #             return candidate
    #
    # def clean_words(name):
    #     name = name.replace("8", "B")
    #     name = name.replace("0", "D")
    #     name = name.replace("6", "G")
    #     name = name.replace("1", "I")
    #     name = name.replace('5', 'S')
    #     return name
    #
    # def adhaar_read_data(textFront, textBack):
    #
    #     textFront = re.sub(r'[^\x00-\x7F]+', '', textFront)  # remove non ascii characters
    #     textBack = re.sub(r'[^\x00-\x7F]+', '', textBack)
    #     resFront = textFront.split()
    #     resBack = textBack.splitlines()
    #
    #     resBack = [i for i in resBack if i != '']
    #
    #     name = None
    #     dob = None
    #     adh = None
    #     sex = None
    #     address = None
    #     temp = ""
    #     #   print(textBack)
    #     try:
    #         address = (
    #             re.search(r'(?:(Address|Adress|Addres|Adres|Addre55):?)(.*(\d{6}))', textBack, re.M | re.S)).group(
    #             2)
    #     except AttributeError:
    #         pass
    #
    #     lines = textFront.split('\n')
    #     for i in range(len(lines)):
    #         if 'DOB' in lines[i]:
    #             name = lines[i - 1]
    #             dob = re.sub(r'.*DOB:?', '', lines[i])
    #             break;
    #     aadhaar_candidates = re.findall(r'\d{4} \d{4} \d{4}', textFront + textBack)
    #     aadharNumber = validate_aadhaar_numbers(aadhaar_candidates)
    #
    #     if 'female' in textFront.lower():
    #         sex = "FEMALE"
    #     else:
    #         sex = "MALE"
    #
    #     try:
    #
    #         name = clean_words(name)
    #         dob = dob.strip()
    #         dob = dob.replace('l', '/')
    #         dob = dob.replace('L', '/')
    #         dob = dob.replace('I', '/')
    #         dob = dob.replace('i', '/')
    #         dob = dob.replace('|', '/')
    #         dob = dob.replace('\"', '/1')
    #         #       dob = dob.replace(":","")
    #         dob = dob.replace(" ", "")
    #
    #         # Cleaning Adhaar number details
    #
    #         for word in resFront:
    #             if len(word) == 4 and word.isdigit():
    #                 aadhar_number = aadhar_number + word + ' '
    #         if len(aadhar_number) >= 14:
    #             print("Aadhar number is :" + aadhar_number)
    #         else:
    #             print("Aadhar number not read")
    #
    #
    #     except:
    #         pass
    #
    #     fathersName = re.search(r"(?::)(.*?)(?:,)", address).group(1)
    #     fathersName.strip()
    #     print(fathersName)
    #     data = {}
    #     data['name'] = name
    #     data['dob'] = dob
    #     data['aadhaar_no'] = aadharNumber
    #     data['sex'] = sex
    #     #                   data['ID Type']='Aadhaar'
    #     data['address'] = address
    #     #                   data['fathers_name'] = fathersName
    #     data['enrollment'] = None
    #     data['vid'] = None
    #     data['phonenumber'] = None
    #     return data
    #
    # def get_details_img(pathF, pathB):
    #
    #     # from google.colab.patches import cv2_imshow
    #     image_front = cv2.imread(pathF)
    #     image_back = cv2.imread(pathB)
    #     angle_front = detect_angle_front(image_front)
    #     #     cv2.imwrite("temp/image_front.png",image_front)
    #     angle_back = detect_angle_back(image_back)
    #     print(angle_front)
    #     print(angle_back)
    #     im = Image.open(pathB)
    #     im = ImageOps.exif_transpose(im)
    #     if angle_back != 0:
    #         im = im.rotate(angle_back, Image.NEAREST, expand=1)
    #     width, height = im.size
    #
    #     left = width / 2
    #     top = 5
    #     right = width
    #     bottom = height
    #
    #     #     im1 = im.crop((left, top, right, bottom))                        #it crops the image, might be a problem
    #
    #     open_cv_image = np.array(im)
    #     dst = cv2.fastNlMeansDenoisingColored(open_cv_image, None, 10, 10, 7, 15)
    #     #     config = ('--psm 1')
    #     extractedbacktext = pytesseract.image_to_string(dst, lang='kan+eng+hin')
    #     extractedbacktext_output = open('output2.txt', 'w', encoding='utf-8')
    #     extractedbacktext_output.write(extractedbacktext)
    #     extractedbacktext_output.close()
    #     file = open('output2.txt', 'r', encoding='utf-8')
    #     extractedbacktext = file.read()
    #     extractedbacktext = ftfy.fix_text(extractedbacktext)
    #     extractedbacktext = ftfy.fix_encoding(extractedbacktext)
    #     print(extractedbacktext)
    #
    #     # from google.colab.patches import cv2_imshow
    #     src = cv2.imread(pathF)
    #     # if angle_front == 0:
    #     #   img = src
    #     # if angle_front == 270:
    #     #   img = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
    #     # if angle_front == 180:
    #     #   img = cv2.rotate(src, cv2.ROTATE_180)
    #     # if angle_front == 90:
    #     #   img = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #     img = src
    #     # cv2_imshow(img)
    #     dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    #     config = ('--psm 3')
    #     extractedfronttext = pytesseract.image_to_string(dst, config=config,
    #                                                      lang='kan+eng+hin')  # try adding other configs
    #     extractedfronttext_output = open('output1.txt', 'w', encoding='utf-8')
    #     extractedfronttext_output.write(extractedfronttext)
    #     extractedfronttext_output.close()
    #     file = open('output1.txt', 'r', encoding='utf-8')
    #     extractedfronttext = file.read()
    #     extractedfronttext = ftfy.fix_text(extractedfronttext)
    #     # print(extractedfronttext)
    #     extractedfronttext = ftfy.fix_encoding(extractedfronttext)
    #     # print(extractedfronttext)
    #     data = adhaar_read_data(extractedfronttext, extractedbacktext)
    #     return data
    # #     try:
    # #         to_unicode = unicode
    # #     except NameError:
    # #         to_unicode = str
    # #     with io.open('info.json', 'w', encoding='utf-8') as outfile:
    # #         data = json.dumps(data, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
    # #         outfile.write(to_unicode(data))
    #
    # #     with open('info.json', encoding = 'utf-8') as data:
    # #         data_loaded = json.load(data)
    #
    # #     print("\n---------- ADHAAR Details ----------")
    # #     print("\nADHAAR Number: ",data_loaded['Adhaar Number'])
    # #     print("\nName: ",data_loaded['Name'])
    # #     print("\nDate Of Birth: ",data_loaded['Date of Birth'])
    # #     print("\nSex: ",data_loaded['Sex'])
    # #     print("\nAddress:",data_loaded['Address'])
    # #     print("\n------------------------------------")
