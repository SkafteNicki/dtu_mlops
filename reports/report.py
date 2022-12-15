# run following commands to install requirements
# pip install click
# pip install markdown

import click
import markdown

@click.group()
def cli():
    pass


@cli.command()
def html():
    with open('README.md', 'r') as file:
        text = file.read()
    text = text[43:] # remove header

    html = markdown.markdown(text)

    with open('report.html', 'w') as newfile:
        newfile.write(html)

@cli.command()
def check():
    with open('README.md', 'r') as file:
        text = file.read()
    text = text[43:] # remove header

    result = []
    per_question = text.split('Answer:')
    for q in per_question:
        if '###' in q:
            result.append(q.split("###")[0])
    result = result[1:]  # remove first section

    question_constrains = [
       None,
       None,
       lambda words: 100 <= words <= 200,
    ]

if __name__ == "__main__":
    cli()
