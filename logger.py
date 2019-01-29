'''
Created on 01/05/2015

@author: Alexandre Yukio Yamashita
'''

class Logger:
    INFO, ERROR, DEBUG = range(3)

    def log(self, category, message):
        '''
        Log message.
        '''
        # print "[" + self._get_category_string(category) + "] " + message

    def _get_category_string(self, category):
        '''
        Get category string.
        '''
        if category == self.ERROR:
            category_string = "ERROR"
        elif category == self.DEBUG:
            category_string = "DEBUG"
        else:
            category_string = "INFO"

        return category_string
