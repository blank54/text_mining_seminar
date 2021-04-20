def naver_crawler():
    f = open("/data/kjyon/crawling/naver_crawling_output/news_contents/contents10000_20000.txt", 'w', encoding='utf-8')

    for n in range(10000,20000,10):
        print(n)
        time.sleep(3)
        url = 'https://search.naver.com/search.naver?&where=news&query=%EA%B1%B4%EC%84%A4&sm=tab_pge&sort=0&photo=0&field=0&reporter_article=&pd=3&ds=20201201&de=20201231&start='+str(n)
        headers = {'User-Agent' : 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36'}

        req = requests.get(url, headers = headers)
    
        cont = req.content
        soup = BeautifulSoup(cont, 'html.parser')

        for urls in soup.select("#main_pack > section.sc_new.sp_nnews._prs_nws > div > div.group_news > ul > li > div > div > div > div > a"):
            try :
                #print(urls["href"])
                if urls["href"].startswith("https://news.naver.com"):
                    #print(urls["href"])
                    
                    news_detail = naver_get_news(urls["href"])
                    print(news_detail)
                    f.write("{}\t{}\t{}\t{}\t{}\n".format(news_detail[0], news_detail[1], news_detail[4], news_detail[3],news_detail[2]))

                    
            except Exception as e:
                print(e)
                continue