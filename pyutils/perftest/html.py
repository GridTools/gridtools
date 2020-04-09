import contextlib
import pathlib
from xml.etree import ElementTree as et

from pyutils import log

_CSS = '''
    table { margin-bottom: 5em; border-collapse: collapse; }
    th { text-align: left; border-bottom: 1px solid black; padding: 0.5em; }
    td { padding: 0.5em; }
    .grid-container { width: 100%; display: grid;
        grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); }
    .grid-item { }
    .good { color: #81b113; background: #dfff79; font-weight: bold; }
    .bad { color: #c23424; background: #ffd0ac; font-weight: bold; }
    .unknown { color: #1f65c2; background: #d5ffff; font-weight: bold; }
    img { width: 100%; }
    html { font-family: sans-serif; }
'''


class Report:
    def __init__(self, filename, title):
        self.filename = pathlib.Path(filename)
        self.data_dir = pathlib.Path(self.filename.stem)
        self.data_dir.mkdir()
        self.data_counter = 0

        self.html = et.Element('html')
        self._init_head()
        self.body = et.SubElement(self.html, 'body')
        header = et.SubElement(self.body, 'h1')
        header.text = title

    def _init_head(self):
        head = et.SubElement(self.html, 'head')
        et.SubElement(head, 'meta', charset='utf-8')
        et.SubElement(head,
                      'meta',
                      name='viewport',
                      content='width=device-width, initial-scale=1.0')
        et.SubElement(head,
                      'link',
                      rel='stylesheet',
                      href=str(self._write_css()))

    def _write_css(self):
        path, rel_path = self.get_data_path('.css')
        with path.open('w') as css_file:
            css_file.write(_CSS)
        log.debug(f'Sucessfully written CSS to {path}')
        return rel_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.write()
        return False

    def write(self):
        et.ElementTree(self.html).write(self.filename,
                                        encoding='utf-8',
                                        method='html')
        log.info(f'Successfully written HTML report to {self.filename}')

    def get_data_path(self, suffix=''):
        path = self.data_dir / f'{self.data_counter:03}{suffix}'
        self.data_counter += 1
        return path, path.relative_to(self.filename.parent)

    @contextlib.contextmanager
    def _section(self, title):
        if title:
            header = et.SubElement(self.body, 'h2')
            header.text = title
        yield self.body

    @contextlib.contextmanager
    def table(self, title=None):
        with self._section(title) as section:
            yield _Table(section)

    @contextlib.contextmanager
    def image_grid(self, title=None):
        with self._section(title) as section:
            yield _Grid(section, self.get_data_path)


class _Table:
    def __init__(self, parent):
        self.html = et.SubElement(parent, 'table')
        self.first = True

    @contextlib.contextmanager
    def row(self):
        yield _TableRow(self.html, self.first)
        self.first = False


class _TableRow:
    def __init__(self, parent, header):
        self.html = et.SubElement(parent, 'tr')
        self.header = header

    def cell(self, text):
        elem = et.SubElement(self.html, 'th' if self.header else 'td')
        elem.text = text
        return elem

    def fill(self, *texts):
        return [self.cell(text) for text in texts]


class _Grid:
    def __init__(self, parent, get_data_path):
        self.html = et.SubElement(parent, 'div', {'class': 'grid-container'})
        self.get_data_path = get_data_path

    def image(self):
        path, rel_path = self.get_data_path('.png')
        et.SubElement(self.html, 'img', {
            'class': 'grid-item',
            'src': str(rel_path)
        })
        return path