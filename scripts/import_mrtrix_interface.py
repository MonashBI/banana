import os.path
import pyparsing as pp
from pprint import pprint

pp.ParserElement().setWhitespaceChars('')
word = pp.Word(pp.printables)
space = pp.White(' ')

st = word + space + word


desc_line = pp.Combine(
    pp.White(' ', exact=5).suppress() +
    pp.OneOrMore(word + pp.Optional(space)) +
    pp.LineEnd())

empty_line = (pp.ZeroOrMore(space) + pp.LineEnd())

desc = pp.Combine(
    desc_line + pp.ZeroOrMore(desc_line | empty_line)).setResultsName('desc')

option_name = (pp.White(' ', exact=2).suppress() +
               pp.Word('-').suppress() +
               word).setResultsName('name')

metavar = pp.Combine(
    pp.ZeroOrMore(pp.White(' ').suppress() + word).setResultsName('metavar'))

option = pp.Group(
    option_name +
    metavar +
    pp.LineEnd().suppress() +
    desc)

parser = pp.ZeroOrMore(option)


def parse(parser, string):
    try:
        p = parser.parseString(string)
        pprint(p)
    except pp.ParseException as e:
        print "Did not parse ({})\n\n:{}".format(string, e)
        p = None
    return p

options_str = """  -type choice
     the registration type. Valid choices are: rigid, affine, nonlinear,
     rigid_affine, rigid_nonlinear, affine_nonlinear, rigid_affine_nonlinear
     (Default: affine_nonlinear)

  -transformed image
     image1 after registration transformed to the space of image2

  -transformed_midway image1_transformed image2_transformed
     image1 and image2 after registration transformed to the midway space

  -mask1 filename
     a mask to define the region of image1 to use for optimisation.

  -mask2 filename
     a mask to define the region of image2 to use for optimisation.
"""

option_str = """  -type choice
     the registration type. Valid choices are: rigid, affine, nonlinear,
     rigid_affine, rigid_nonlinear, affine_nonlinear, rigid_affine_nonlinear
     (Default: affine_nonlinear)"""

option_head_str = """  -type choice"""

st_str = 'type choice'

desc_str = """     the registration type. Valid choices are: rigid, affine, nonlinear,
     rigid_affine, rigid_nonlinear, affine_nonlinear, rigid_affine_nonlinear
     (Default: affine_nonlinear)"""

metavar_str = ' image1_transformed image2_transformed'

option_head_str2 = '  -transformed_midway image1_transformed image2_transformed'

if __name__ == '__main__':
    example_fname = os.path.join(os.path.dirname(__file__),
                                 'mrregister-help.txt')
#     parse(option_head, option_head_str)
#     parse(st, st_str)
#     parse(desc, desc_str)
#     parse(option, option_str)
    p = parse(parser, options_str)
#     p = parse(metavar, metavar_str)
#     p = parse(option, option_head_str2)
    print('done')
