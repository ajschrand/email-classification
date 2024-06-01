import pandas
import os
import re
import concurrent.futures

def parseEmailString(emailString):
    (header, body) = emailString.split('\n\n', 1)

    # remove newlines from header
    header = header.replace('\n', ' ')

    wordsToSplit = ['Message-ID: ', 'Date: ', 'From: ', 'To: ', 'Subject: ', 'Cc: ', 'Mime-Version: ', 'Content-Type: ',
        'Content-Transfer-Encoding: ', 'Bcc: ', 'X-From: ', 'X-To: ', 'X-cc: ', 'X-bcc: ', 'X-Folder: ', 'X-Origin: ',
        'X-FileName: ', '\n\n']

    headers = []
    # get the content between the first and second words to split, then between the second and third, etc.
    for i in range(len(wordsToSplit) - 1):
        content = re.search(wordsToSplit[i] + '(.*)' + wordsToSplit[i + 1], header)

        if content is not None:
            headers.append(content.group(1))
        else:
            headers.append('')

    # split the body
    if '-----' in body:
        body = body.split('-----')[0]

    return (headers, body.strip())


def parseFolder(name):
    # make a new dataframe with columns: content, messageID, date, sender, 
    # recipient, subject, mimeVersion, contentType, contentTransferEncoding, xFrom, 
    # xTo, xcc, xBcc, xFolder, xOrigin, xFileName
    df = pandas.DataFrame(columns=['file', 'content', 'message-ID', 'date', 'sender', 
    'recipient', 'subject', 'cc', 'mimeVersion', 'contentType', 'contentTransferEncoding', 
    'bcc', 'xFrom', 'xTo', 'xcc', 'xBcc', 'xFolder', 'xOrigin', 'xFileName'])

    print(f'Parsing folder {name}')
    for folder in os.listdir(f'project/maildir/{name}'):
        if not os.path.isdir(f'project/maildir/{name}/{folder}'):
            continue

        for file in os.listdir(f'project/maildir/{name}/{folder}'):
            if os.path.isdir(f'project/maildir/{name}/{folder}/{file}'):
                continue

            with open(f'project/maildir/{name}/{folder}/{file}', 'r') as f:
                emailString = f.read()
                parsed = parseEmailString(emailString)
                headers = parsed[0]
                body = parsed[1]

                currentLength = len(df) - 1

                # make a new row in the dataframe. 
                df.loc[currentLength, 'file'] = file

                # add the body of the email to the dataframe
                df.loc[currentLength, 'content'] = body

                # add the headers to the dataframe
                for i in range(len(headers)):
                    df.loc[currentLength, df.columns[i + 2]] = headers[i]

    return df

results = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    for name in os.listdir('project/maildir'):
        if not os.path.isdir(f'project/maildir/{name}'):
            continue

        results.append(executor.submit(parseFolder, name).result())

df = pandas.concat(results)

# drop where content is empty
df = df[df['content'] != '']

df.to_csv('project/parsed_emails.csv', index=False)