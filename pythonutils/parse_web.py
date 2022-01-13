def get_link_html(cur_link):
    path_to_phantomJS='/media/anantgupta/ubuntu_1471/ProjectsCopy/office_jp/phantomjs'
    browser = webdriver.PhantomJS(executable_path = path_to_phantomJS, 
                                 service_log_path=os.path.devnull,
                                 service_args=['--ignore-ssl-errors=true'])
    browser.get(cur_link)
    time.sleep(1)
    page_contents=browser.page_source
    browser.close()
    browser.quit()
    return page_contents

cur_link=''

def get_links_data(search_string, link_data, num_results=1):
    headers = {'user-agent': 'my-app/0.0.1'}
    mainLink='https://duckduckgo.com/html/?q={}?b='.format(search_string.replace(' ','+'))
    #mainLink='https://www.ecosia.org/search?q={}'.format(searchString.replace(' ','+'))
    #mainLink='https://en.wikipedia.org/w/index.php?search={}'.format(searchString.replace(' ','+'))
    print("the search link getting executed is {0}".format(mainLink))
    response = requests.get(mainLink, headers = headers)
    #soup = bs(response.text, parser="html.parser")
    soup = bs(response.text)
    soup = soup.find("div", attrs={"id": "links", "class": "results"})
    cur_result_count=0
    for cur_link in soup.findChildren('a', attrs={"class": "result__a"}):
        print(cur_link)
        href_link = cur_link.get('href')
        href_link = href_link.replace('//duckduckgo.com/l/?uddg=', '')
        href_link = href_link[0:href_link.find('&rut')]
        href_link = href_link.replace('%3A',':').replace('%2F','/').replace('%2D','-')
        print("the final href_link is {0}".format(href_link))
        link_data[href_link] = get_link_html(href_link)
        print("Completed getting data for {0}".format(href_link))
        cur_result_count = cur_result_count + 1
        if cur_result_count==num_results:
            break
    return link_data

full_data= {}
company_list=[]
for cur_company in company_list:
    search_string='{0} Corp climate_change'.format(cur_company)
    link_data = {}
    link_data = get_links_data(search_string, link_data)
    full_data[cur_company] = link_data
