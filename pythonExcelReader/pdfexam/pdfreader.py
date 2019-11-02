import PyPDF2
pdfFileObj = open('SOUTH CAMPUS NOV 2019.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
for m in range(pdfReader.numPages):
    pageObj = pdfReader.getPage(m)
    flag = False
    for q in range(len(pageObj.extractText().split("\n"))):
        line = pageObj.extractText().split("\n")[q]
        if "-----" in line and flag == False:
            flag = True
            continue
        if "****" in line or "sum" in line:
            flag = False
            q += 1
        if flag == True:
            print(pageObj.extractText().split("\n")[q]) 
                  
