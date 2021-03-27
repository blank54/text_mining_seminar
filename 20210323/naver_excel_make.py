def naver_excel_make():
    data = pd.read_csv(RESULT_PATH+'contents10000_20000.txt', sep='\t', error_bad_lines=False)
    data.columns = ['title','date','company','link','contents']
    #print(data)
    #xlsx_outputFileName = '%s-%s-%s %sΩ√ %s∫– %s√  result.xlsx' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    #xlsx_name = 'result' + '.xlsx'
    #data.to_excel(RESULT_PATH+xlsx_outputFileName, encoding='utf-8')
    data.to_excel(RESULT_PATH+"result10000_20000.xlsx", encoding='utf-8')

