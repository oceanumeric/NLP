## L1 NLP Setup, Web Scraping and OLS 

### Developer Tools 

* python: https://www.python.org/
* vscode: https://code.visualstudio.com/
* wsl: https://docs.microsoft.com/en-us/windows/wsl/install
* git: https://git-scm.com/


Vscode Preference Settings (`Ctrl+Shift+p`)

```
{
    "workbench.colorCustomizations": {
 
        "editor.background": "#203830",
        "activityBar.background": "#202A2F",
        "sideBar.background": "#202A2F",
        "titleBar.activeBackground": "#3D3C3D",
        "tab.activeBackground": "#263238"
       
    },
    "editor.fontSize": 14,
    "editor.rulers": [80],
}
```


### Web Scraping

Way Back Machine: https://web.archive.org/

Protocol: a set of rules governing the exchange or transmission of data 
between devices.

Hypertext Transfer Protocol (HTTP) is an application-layer protocol for 
transmitting hypermedia documents, such as HTML. It was designed for 
communication between web browsers and web servers, but it can also be 
used for other purposes.


http://data.pr4e.org/page1.htm: protocol + host + document (data)

```bash
telnet data.pr4e.org 80 
```
    Trying 192.241.136.170...
    Connected to data.pr4e.org.
    Escape character is '^]'.
```
GET http://data.pr4e.org/page1.htm HTTP/1.0
```

    HTTP/1.1 200 OK
    Date: Fri, 01 Oct 2021 10:20:28 GMT
    Server: Apache/2.4.18 (Ubuntu)
    Last-Modified: Mon, 15 May 2017 11:11:47 GMT
    ETag: "80-54f8e1f004857"
    Accept-Ranges: bytes
    Content-Length: 128
    Cache-Control: max-age=0, no-cache, no-store, must-revalidate
    Pragma: no-cache
    Expires: Wed, 11 Jan 1984 05:00:00 GMT
    Connection: close
    Content-Type: text/html

```
<h1>The First Page</h1>
<p>
If you like, you can switch to the 
<a href="http://data.pr4e.org/page2.htm">
Second Page</a>.
</p>
Connection closed by foreign host.
```

### HTML, CSS, Javascript

Link: https://codepen.io/rcyou/pen/QEObEk

```html
<h2> Hello, world </h2>

<button type="button">
Click me to display Date and Time.</button>

<p id="demo"></p>

<a href="https://www.w3schools.com">Visit W3Schools</a>
```

```css
h2 {
    color:red
}
```

```javascript
$('button').on('click', function() {
	document.getElementById('demo').innerHTML = Date();
});
```

### requests and get

```python
import requests 


page = requests.get('https://www.economist.com/')
page.text
```

### Beautiful Soup

```python
import requests 
from bs4 import BeautifulSoup


page = requests.get('https://www.economist.com/')
soup = BeautifulSoup(page.content, 'lxml')
soup.find('h3', id='edition-headline')
```

__class__ and __id__ attribute

```python
SoupObject.find('name', _class='class_attribute', id='id_attribute')
```

### Learning by doing

Uni Link https://search.sjtu.edu.cn/search-news/search






