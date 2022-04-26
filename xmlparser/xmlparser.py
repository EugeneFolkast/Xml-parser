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
        j = 0

        for element in self.files:
            # Образуем дерево тэгов из xml файлов
            self.tree = ET.parse('D:/gits/Unettestproj/source/xml/' + element).getroot()



            buf1 = element.split('.')
            buf2 = self.imgFilse[i].split('.')
            buf3 ='D:/gits/Unettestproj/source/xml/' + buf1[0]
            buf4 ='D:/gits/Unettestproj/source/xml/' + buf2[0]

            if (buf3) != buf4:
                continue


            for tag in self.tree.findall('object'):
                # Получаем изображение для обработки opencv в соответствии с именем подходящего xml файла
                self.img = cv2.imread('D:/gits/Unettestproj/source/sourceimg/' + self.imgFilse[i])

                objectName = tag[0].text

                self.dataListForOne.append(objectName)

                for child in tag.find('bndbox'):
                    self.dataListForOne.append(child.text)

                self.dataList.append(self.dataListForOne.copy())

                cv2.rectangle(self.img, (int(self.dataListForOne[1]), int(self.dataListForOne[2])),
                                        (int(self.dataListForOne[3]), int(self.dataListForOne[4])), 1)


                stencil = numpy.zeros(self.img.shape).astype(self.img.dtype)
                contours = [numpy.array([[int(self.dataListForOne[1]), int(self.dataListForOne[2])],
                                         [int(self.dataListForOne[3]), int(self.dataListForOne[2])],
                                         [int(self.dataListForOne[3]), int(self.dataListForOne[4])],
                                         [int(self.dataListForOne[1]), int(self.dataListForOne[4])]])]

                cv2.fillPoly(stencil, contours, color=[255, 255, 255])
                result = cv2.bitwise_and(self.img, stencil)

                # обрезаем картинку по нужным координатам
                result = result[int(self.dataListForOne[2]):int(self.dataListForOne[4]),
                         int(self.dataListForOne[1]):int(self.dataListForOne[3])]


                # Сохраняем обработанное изображение
                cv2.imwrite('D:/gits/Unettestproj/source/proceedimgdir/' + objectName + '/' + self.imgFilse[i], result)

                j += 1

                self.dataListForOne.clear()

            i += 1
            j = 0


if __name__ == '__main__':
    xmlRun = Xmlparser()

    xmlRun.printRootNode()

