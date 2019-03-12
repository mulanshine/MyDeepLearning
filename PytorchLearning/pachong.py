# -*- coding:utf-8 -*-  
import urllib  
import sys  
import chardet  
import re  
  
  
def get_html(url):  
    page = urllib.urlopen(url)  
    content = page.read()  
    typeEncode = sys.getfilesystemencoding()  
    infoencode = chardet.detect(content).get('encoding','utf-8')  
    html = content.decode(infoencode,'ignore').encode(typeEncode)  
    return html      #提取html页面，编码已经转换  
  #表达式中只有一个括号时，findall只会返回括号的内容
def get_img(html):  
    reg = r'src="(http://imgsrc.baidu.com/forum/w.*?\.jpg)"'   
    imgre = re.compile(reg)  
    imglist = re.findall(imgre, html)   
    i = 0  
    for imgurl in imglist:  
        print(imgurl)  
        urllib.urlretrieve(imgurl, 'C:/Users/lijiong/Desktop/new/%s.jpg'%i)  
        i+=1  
          
html = get_html('http://tieba.baidu.com/p/3837885162')  
get_img(html)  