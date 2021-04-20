def naver_get_news(n_url):
    news_detail = []
    time.sleep(3)

    headers = {'User-Agent' : 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36'}

    req2 = requests.get(n_url, headers = headers)
    soup2 = BeautifulSoup(req2.content, 'html.parser')

    title = soup2.select('h3#articleTitle')[0].text #���ȣ�� h3#articleTitle �� ���� ù��° �׷츸 �������ڴ�.
    news_detail.append(title)

    pdate = soup2.select('.t11')[0].get_text()[:11]
    news_detail.append(pdate)

    _text = soup2.select('#articleBodyContents')[0].get_text().replace('\n', " ")
    text = _text.replace('\t', '')
    text2 = text.replace('// flash ������ ��ȸ�ϱ� ���� �Լ� �߰� function _flash_removeCallback() {}', "")
    news_detail.append(text2.strip())



    news_detail.append(n_url)

    company = soup2.select('#footer address')[0].a.get_text()
    news_detail.append(company)
    
    return news_detail