import xml.etree.ElementTree as ET
import cv2
import numpy
import os


class Xmlparser:

    def __init__(self):
        """
        Инициализируем необходимые нам данные

        """


        self.dataListForOne = list()
        self.dataList = list()

        # Директория с xml-файлами
        self.xmlDirectory = "D:/gits/Unettestproj/source/xml/"
        # Директория исходных изображений
        self.sourceImgDirectory = "D:/gits/Unettestproj/source/sourceimg/"
        # Директория для сохранения изображений после обработки скриптом
        self.proceedImgDirectory = "D:/gits/Unettestproj/source/proceedimgdir/"

        # Получаем список файлов в xml директории
        self.files = os.listdir(self.xmlDirectory)
        # Получаем список файлов в директории исходных изображений
        self.imgFilse = os.listdir(self.sourceImgDirectory)

        self.img = cv2.imread('D:/gits/Unettestproj/source/sourceimg/i (1).jpg')

        self.CLASSES = {'nothing':0 ,'armchair':1, 'chair':2,
                        'coach':3, 'dressers_and_cabinets_with_drawers':4,
                        'floor_cabinet':5, 'mirror':6,
                        'office_chair':7, 'shelving_and_bookcases':8,
                        'table_lamps':9, 'wall_shelves':10, 'table_lmps':9,
                        'wall_shelves_for_books':11, 'writing_desk':12, 'table':13}

        self.COLORS = {0:(0, 255, 0), 1:(128, 255, 0),
                       2:(196, 255, 0), 3:(255, 255, 0),
                       4:(255, 188, 0), 5:(255, 119, 0),
                       6:(255, 150, 0), 7:(255, 255, 0),
                       8:(255, 50, 0), 9:(255, 80, 0),
                       10:(255, 130, 0), 11:(255, 170, 0),
                       12:(255, 200, 0), 13:(255, 255, 0)}

    def printRootNode(self):
        """
        Метод поочерёдно обрабатывает каждый xml файл из директории
        Открывая xml файл, получаем список всех тэгов object каждого xml файла,
        Для каждого тэга object уже заранее созданы соответствующие директории в папке
        обработанных изображений.
        В соответствии с количеством тэгов Object и их типом - отрисовывается область нахождения объекта на изображении,
        и сохраняется файл в соответствующую папку объекта.
        :return:
        """
        i = 0

        for element in self.files:
            # Образуем дерево тэгов из xml файлов
            self.tree = ET.parse('D:/gits/Unettestproj/source/xml/' + element).getroot()



            buf1 = element.split('.')
            buf2 = self.imgFilse[i].split('.')
            buf3 ='D:/gits/Unettestproj/source/xml/' + buf1[0]
            buf4 ='D:/gits/Unettestproj/source/xml/' + buf2[0]

            if (buf3) != buf4:
                continue

            self.img = cv2.imread('D:/gits/Unettestproj/source/sourceimg/' + self.imgFilse[i])

            rows, cols = self.img.shape[:2]

            contours1 = [numpy.array([[0, 0],
                                     [cols, 0],
                                    [cols, rows],
                                     [0, rows]])]

            cv2.fillPoly(self.img, contours1, (0, 0, 0))

            for tag in self.tree.findall('object'):

                if len(tag[0].text) < 5:
                    continue

                objectName = tag[0].text

                self.dataListForOne.append(objectName)

                for child in tag.find('bndbox'):
                    self.dataListForOne.append(child.text)

                self.dataList.append(self.dataListForOne.copy())

                colorClass = self.CLASSES[objectName]
                objectColor = self.COLORS[colorClass]

                cv2.rectangle(self.img, (int(self.dataListForOne[1]), int(self.dataListForOne[2])),
                                        (int(self.dataListForOne[3]), int(self.dataListForOne[4])), objectColor, 1)


                contours = [numpy.array([[int(self.dataListForOne[1]), int(self.dataListForOne[2])],
                                         [int(self.dataListForOne[3]), int(self.dataListForOne[2])],
                                         [int(self.dataListForOne[3]), int(self.dataListForOne[4])],
                                         [int(self.dataListForOne[1]), int(self.dataListForOne[4])]])]

                cv2.fillPoly(self.img, contours, objectColor)

                self.dataListForOne.clear()

            i += 1

            self.img = cv2.resize(self.img, (512, 512))

            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

            # Сохраняем обработанное изображение
            cv2.imwrite('D:/gits/Unettestproj/source/proceedimgdir/data/' + self.imgFilse[i], self.img)


if __name__ == '__main__':
    xmlRun = Xmlparser()

    xmlRun.printRootNode()

