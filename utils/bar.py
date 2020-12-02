# -*- coding: utf-8 -*-

# Copyright (c) 2012 Giorgos Verigakis <verigak@gmail.com>
#
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from __future__ import unicode_literals
from __future__ import print_function
from . import Progress


HIDE_CURSOR = '\x1b[?25l'
SHOW_CURSOR = '\x1b[?25h'


class WriteMixin(object):
    hide_cursor = False

    def __init__(self, message=None, **kwargs):
        super(WriteMixin, self).__init__(**kwargs)
        self._width = 0
        if message:
            self.message = message

        if self.file.isatty():
            if self.hide_cursor:
                print(HIDE_CURSOR, end='', file=self.file)
            print(self.message, end='', file=self.file)
            self.file.flush()

    def write(self, s):
        if self.file.isatty():
            b = '\b' * self._width
            c = s.ljust(self._width)
            print(b + c, end='', file=self.file)
            self._width = max(self._width, len(s))
            self.file.flush()

    def finish(self):
        if self.file.isatty() and self.hide_cursor:
            print(SHOW_CURSOR, end='', file=self.file)


class WritelnMixin(object):
    hide_cursor = False

    def __init__(self, message=None, **kwargs):
        super(WritelnMixin, self).__init__(**kwargs)
        if message:
            self.message = message

        if self.file.isatty() and self.hide_cursor:
            print(HIDE_CURSOR, end='', file=self.file)

    def clearln(self):
        if self.file.isatty():
            print('\r\x1b[K', end='', file=self.file)

    def writeln(self, line):
        if self.file.isatty():
            self.clearln()
            print(line, end='', file=self.file)
            self.file.flush()

    def finish(self):
        if self.file.isatty():
            print(file=self.file)
            if self.hide_cursor:
                print(SHOW_CURSOR, end='', file=self.file)


from signal import signal, SIGINT
from sys import exit


class SigIntMixin(object):
    """Registers a signal handler that calls finish on SIGINT"""

    def __init__(self, *args, **kwargs):
        super(SigIntMixin, self).__init__(*args, **kwargs)
        signal(SIGINT, self._sigint_handler)

    def _sigint_handler(self, signum, frame):
        self.finish()
        exit(0)



class Bar(WritelnMixin, Progress):
    width = 32
    message = ''
    suffix = '%(index)d/%(max)d'
    bar_prefix = ' |'
    bar_suffix = '| '
    empty_fill = ' '
    fill = '#'
    hide_cursor = True

    def update(self):
        filled_length = int(self.width * self.progress)
        empty_length = self.width - filled_length

        message = self.message % self
        bar = self.fill * filled_length
        empty = self.empty_fill * empty_length
        suffix = self.suffix % self
        line = ''.join([message, self.bar_prefix, bar, empty, self.bar_suffix,
                        suffix])
        self.writeln(line)


class ChargingBar(Bar):
    suffix = '%(percent)d%%'
    bar_prefix = ' '
    bar_suffix = ' '
    empty_fill = '∙'
    fill = '█'


class FillingSquaresBar(ChargingBar):
    empty_fill = '▢'
    fill = '▣'


class FillingCirclesBar(ChargingBar):
    empty_fill = '◯'
    fill = '◉'


class IncrementalBar(Bar):
    phases = (' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█')

    def update(self):
        nphases = len(self.phases)
        filled_len = self.width * self.progress
        nfull = int(filled_len)                      # Number of full chars
        phase = int((filled_len - nfull) * nphases)  # Phase of last char
        nempty = self.width - nfull                  # Number of empty chars

        message = self.message % self
        bar = self.phases[-1] * nfull
        current = self.phases[phase] if phase > 0 else ''
        empty = self.empty_fill * max(0, nempty - len(current))
        suffix = self.suffix % self
        line = ''.join([message, self.bar_prefix, bar, current, empty,
                        self.bar_suffix, suffix])
        self.writeln(line)


class PixelBar(IncrementalBar):
    phases = ('⡀', '⡄', '⡆', '⡇', '⣇', '⣧', '⣷', '⣿')


class ShadyBar(IncrementalBar):
    phases = (' ', '░', '▒', '▓', '█')