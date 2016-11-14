"""
sct_report configuration file
"""

# name of the subdirectory where sct_report will be saved
reportDirName = 'sct_report'
reportConfigFileName = '.config.json'

# images format returned by sct_tool
imagesExt = '*.png'

# subdirectory where sct_report images will  be save
imagesDirName = 'img'

# html_templates
templatesDirName = 'html_templates'
indexTemplate = 'index.html'
constrasteToolTemplate = 'contrast_tool.html'
menuTemplate = 'menu.html'

#used only in developpement TODO:remove me in production
exempleDateDir= 'sct_exemple_data'

logo = "sct_logo.png"

#assets needed by the templates
assetsDirName =  'assets'
# css, js files  needed
requiredFilesExt = ['*.js',"*.css"]



