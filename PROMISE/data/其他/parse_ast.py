import javalang
import csv
import os

projects = [
    'ant',
    'camel',
    'ivy',
    'jedit',
    'log4j',
    'lucene',
    'poi',
    'synapse',
    'xalan',
    'xerces',
    "velocity"
]

versions = {
    # 'ant':['1.3', '1.4','1.5', '1.6', '1.7'],
    'ant': ['1.5', '1.6', '1.7'],
    'camel': ['1.0', '1.2', '1.4', '1.6'],
    'jedit': ['3.2', '4.0', '4.1'],
    'log4j': ['1.0', '1.1', '1.2'],
    'lucene': ['2.0', '2.2', '2.4'],
    'xalan': ['2.4', '2.5', '2.6', '2.7'],
    'xerces': ['1.2', '1.3'],
    'ivy': ['1.4', '2.0'],
    'synapse': ['1.0', '1.1', '1.2'],
    'poi': ['1.5', '2.0', '2.5', '3.0'],
    'velocity': ['1.4', '1.5', '1.6']
}

# max_lengths = {
#     'ant': 500,
#     'camel':900,
#     'ivy':1500,
#     'jedit':2500,
#     'log4j':1200,
#     'lucene':1500,
#     'poi':1800,
#     'synapse':1200,
#     'xalan':2000,
#     'xerces':2000
# }

dict_dir = {"ant": ["src/main", "proposal/sandbox/junit/src/main"],
            "camel": ["camel-core/src/main/java", 'components/camel-activemq/src/main/java',
                      'components/camel-bam/src/main/java', 'components/camel-cxf/src/main/java',
                      'components/camel-ftp/src/main/java', 'components/camel-http/src/main/java',
                      'components/camel-irc/src/main/java', 'components/camel-jaxb/src/main/java',
                      'components/camel-jbi/src/main/java', 'components/camel-jms/src/main/java',
                      'components/camel-josql/src/main/java', 'components/camel-jpa/src/main/java',
                      'components/camel-mail/src/main/java', 'components/camel-mina/src/main/java',
                      'components/camel-quartz/src/main/java', 'components/camel-rmi/src/main/java',
                      'components/camel-saxon/src/main/java', 'components/camel-script/src/main/java',
                      'components/camel-spring/src/main/java', 'components/camel-xmpp/src/main/java',
                      'components/pom.xml/src/main/java', 'components/camel-amqp/src/main/java',
                      'components/camel-atom/src/main/java', 'components/camel-groovy/src/main/java',
                      'components/camel-ibatis/src/main/java', 'components/camel-jdbc/src/main/java',
                      'components/camel-jetty/src/main/java', 'components/camel-jhc/src/main/java',
                      'components/camel-jing/src/main/java', 'components/camel-juel/src/main/java',
                      'components/camel-msv/src/main/java', 'components/camel-ognl/src/main/java',
                      'components/camel-osgi/src/main/java', 'components/camel-ruby/src/main/java',
                      'components/camel-stringtemplate/src/main/java',
                      'components/camel-velocity/src/main/java', 'components/camel-xmlbeans/src/main/java',
                      'components/camel-csv/src/main/java', 'components/camel-flatpack/src/main/java',
                      'components/camel-hamcrest/src/main/java', 'components/camel-jcr/src/main/java',
                      'components/camel-jxpath/src/main/java', 'components/camel-scala/src/main/java',
                      'components/camel-spring-integration/src/main/java',
                      'components/camel-sql/src/main/java', 'components/camel-stream/src/main/java',
                      'components/camel-supercsv/src/main/java', 'components/camel-swing/src/main/java',
                      'components/camel-testng/src/main/java', 'components/camel-uface/src/main/java',
                      'components/camel-xstream/src/main/java', 'components/camel-freemarker/src/main/java',
                      'components/camel-guice/src/main/java', 'components/camel-hl7/src/main/java',
                      'components/camel-jmxconnect/src/main/java', 'components/camel-jt400/src/main/java',
                      'components/camel-ldap/src/main/java', 'components/camel-rest/src/main/java',
                      'components/camel-restlet/src/main/java', 'components/camel-tagsoup/src/main/java'],
            "jedit": [""],
            "log4j": ["src/java"],
            "lucene": ["src/java"],
            "poi": ["src/java"],
            "velocity": ["src/java"],
            "xalan": ["src", "compat_src"],
            "xerces": ["src"],
            "ivy": ["src/java"],
            "synapse": ["modules/core/src/main/java"]}


def parse_ast_ori(path):
    data = open(path, encoding='utf-8').read()
    tree = javalang.parse.parse(data)
    res = []
    for path, node in tree:
        # res.append(node)
        pattern = javalang.tree.ReferenceType
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('Reference Type ' + node.name)
        pattern = javalang.tree.MethodInvocation
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('Method Invocation ' + node.member)
        pattern = javalang.tree.MethodDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('Method Declaration ' + node.name)
        pattern = javalang.tree.TypeDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('Type Declaration ' + node.name)
        pattern = javalang.tree.ClassDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('Class Declaration ' + node.name)
        pattern = javalang.tree.EnumDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('Enum Declaration ' + node.name)
        pattern = javalang.tree.IfStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("if")
        pattern = javalang.tree.WhileStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("while")
        pattern = javalang.tree.DoStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("do")
        pattern = javalang.tree.ForStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("for")
        pattern = javalang.tree.AssertStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("assert")
        pattern = javalang.tree.BreakStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("break")
        pattern = javalang.tree.ContinueStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("continue")
        pattern = javalang.tree.ReturnStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("return")
        pattern = javalang.tree.ThrowStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("throw")
        pattern = javalang.tree.SynchronizedStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("synchronized")
        pattern = javalang.tree.TryStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("try")
        pattern = javalang.tree.SwitchStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("switch")
        pattern = javalang.tree.BlockStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("block")
        pattern = javalang.tree.StatementExpression
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("statement expression")
        pattern = javalang.tree.TryResource
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("try resource")
        pattern = javalang.tree.CatchClause
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("catch clause")
        pattern = javalang.tree.CatchClauseParameter
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("catch clause parameter")
        pattern = javalang.tree.SwitchStatementCase
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("switch statement case")
        pattern = javalang.tree.ForControl
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("for control")
        pattern = javalang.tree.EnhancedForControl
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("enhanced for control")

    return ' '.join(res)


def parse_ast2(path):
    data = open(path, encoding='utf-8').read()
    tree = javalang.parse.parse(data)
    res = []
    for path, node in tree:
        # res.append(node)
        pattern = javalang.tree.ReferenceType
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('Reference Type ' + node.name)
        pattern = javalang.tree.MethodInvocation
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('Method Invocation ' + node.member)
        pattern = javalang.tree.MethodDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('Method Declaration ' + node.name)
        pattern = javalang.tree.TypeDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('Type Declaration ' + node.name)
        pattern = javalang.tree.ClassDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('Class Declaration ' + node.name)
        pattern = javalang.tree.EnumDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('Enum Declaration ' + node.name)
        pattern = javalang.tree.IfStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("if statement")
        pattern = javalang.tree.WhileStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("while statement")
        pattern = javalang.tree.DoStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("do statement")
        pattern = javalang.tree.ForStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("for statement")
        pattern = javalang.tree.AssertStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("assert statement")
        pattern = javalang.tree.BreakStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("break statement")
        pattern = javalang.tree.ContinueStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("continue statement")
        pattern = javalang.tree.ReturnStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("return statement")
        pattern = javalang.tree.ThrowStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("throw statement")
        pattern = javalang.tree.SynchronizedStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("synchronized statement")
        pattern = javalang.tree.TryStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("try statement")
        pattern = javalang.tree.SwitchStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("switch statement")
        pattern = javalang.tree.BlockStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("block statement")
        pattern = javalang.tree.StatementExpression
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("statement expression")
        pattern = javalang.tree.TryResource
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("try resource")
        pattern = javalang.tree.CatchClause
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("catch clause")
        pattern = javalang.tree.CatchClauseParameter
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("catch clause parameter")
        pattern = javalang.tree.SwitchStatementCase
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("switch statement case")
        pattern = javalang.tree.ForControl
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("for control")
        pattern = javalang.tree.EnhancedForControl
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("enhanced for control")

    return ' '.join(res)


def parse_ast_without_prefix(path):
    data = open(path, encoding='utf-8').read()

    tree = javalang.parse.parse(data)
    res = []
    for path, node in tree:
        # res.append(node)
        pattern = javalang.tree.ReferenceType
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append(node.name)
        pattern = javalang.tree.MethodInvocation
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append(node.member)
        pattern = javalang.tree.MethodDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append(node.name)
        pattern = javalang.tree.TypeDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append(node.name)
        pattern = javalang.tree.ClassDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append(node.name)
        pattern = javalang.tree.EnumDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append(node.name)
        pattern = javalang.tree.IfStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("if statement")
        pattern = javalang.tree.WhileStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("while statement")
        pattern = javalang.tree.DoStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("do statement")
        pattern = javalang.tree.ForStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("for statement")
        pattern = javalang.tree.AssertStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("assert statement")
        pattern = javalang.tree.BreakStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("break statement")
        pattern = javalang.tree.ContinueStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("continue statement")
        pattern = javalang.tree.ReturnStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("return statement")
        pattern = javalang.tree.ThrowStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("throw statement")
        pattern = javalang.tree.SynchronizedStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("synchronized statement")
        pattern = javalang.tree.TryStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("try statement")
        pattern = javalang.tree.SwitchStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("switch statement")
        pattern = javalang.tree.BlockStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("block statement")
        pattern = javalang.tree.StatementExpression
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("statement expression")
        pattern = javalang.tree.TryResource
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("try resource")
        pattern = javalang.tree.CatchClause
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("catch clause")
        pattern = javalang.tree.CatchClauseParameter
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("catch clause parameter")
        pattern = javalang.tree.SwitchStatementCase
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("switch statement case")
        pattern = javalang.tree.ForControl
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("for control")
        pattern = javalang.tree.EnhancedForControl
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("enhanced for control")

    return ' '.join(res)


def get_source(path):
    data = open(path, encoding='utf-8').read()
    tree = javalang.parse.parse(data)
    tokens = javalang.parse.tokenize(data)
    tokenlist = []
    tokenstr = ""
    len = 0
    for item in tokens:
        len = len + 1
        tokenlist.append(item.value)
        tokenstr = tokenstr + str(item.value) + " "
    # tokenstr=tokenstr.strip()
    # todo 不要print了，会unicode导致抛出异常
    # print(str(len) + " " + tokenstr)
    # try:
    #
    # except(UnicodeEncodeError):
    #     print("UnicodeEncodeError")
    return tokenstr


def get_source_code(path):
    content = ""
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            content = content + " " + line.rstrip('\n')
    print(content)
    return content


def get_mix_arr(arr1, arr2):
    pass
    res = []
    for index in range(0, min(len(arr1), len(arr2))):
        # 总是先放true的
        res.append(arr1[index])
        res.append(arr2[index])
    return res


def parse_ast(path):
    data = open(path, encoding='utf-8').read()
    tree = javalang.parse.parse(data)
    tokens = javalang.parse.tokenize(data)

    tokenlist = []
    for item in tokens:
        tokenlist.append(item.value)
    # print(str(tokenlist))

    res = []
    for path, node in tree:
        # res.append(node)
        pattern = javalang.tree.ReferenceType
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('ReferenceType_' + node.name)
        pattern = javalang.tree.MethodInvocation
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('MethodInvocation_' + node.member)
        pattern = javalang.tree.MethodDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('MethodDeclaration_' + node.name)
        pattern = javalang.tree.TypeDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('TypeDeclaration_' + node.name)
        pattern = javalang.tree.ClassDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('ClassDeclaration_' + node.name)
        pattern = javalang.tree.EnumDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('EnumDeclaration_' + node.name)
        pattern = javalang.tree.IfStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("ifstatement")
        pattern = javalang.tree.WhileStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("whilestatement")
        pattern = javalang.tree.DoStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("dostatement")
        pattern = javalang.tree.ForStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("forstatement")
        pattern = javalang.tree.AssertStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("assertstatement")
        pattern = javalang.tree.BreakStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("breakstatement")
        pattern = javalang.tree.ContinueStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("continuestatement")
        pattern = javalang.tree.ReturnStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("returnstatement")
        pattern = javalang.tree.ThrowStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("throwstatement")
        pattern = javalang.tree.SynchronizedStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("synchronizedstatement")
        pattern = javalang.tree.TryStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("trystatement")
        pattern = javalang.tree.SwitchStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("switchstatement")
        pattern = javalang.tree.BlockStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("blockstatement")
        pattern = javalang.tree.StatementExpression
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("statementexpression")
        pattern = javalang.tree.TryResource
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("tryresource")
        pattern = javalang.tree.CatchClause
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("catchclause")
        pattern = javalang.tree.CatchClauseParameter
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("catchclauseparameter")
        pattern = javalang.tree.SwitchStatementCase
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("switchstatementcase")
        pattern = javalang.tree.ForControl
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("forcontrol")
        pattern = javalang.tree.EnhancedForControl
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("enhancedforcontrol")

    return ' '.join(res)


if __name__ == "__main__":
    all_buggy = []
    all = []
    all_true = []
    all_mix = []
    all_sequence_and_label_file = open('./sequence_and_label/all.txt', 'w+', encoding='utf-8')
    all_buggy_sequence_and_label_file = open('./sequence_and_label/all_buggy.txt', 'w+', encoding='utf-8')
    all_true_sequence_and_label_file = open('./sequence_and_label/all_true.txt', 'w+', encoding='utf-8')
    all_mix_sequence_and_label_file = open('./sequence_and_label/all_mix.txt', 'w+', encoding='utf-8')

    for project in projects:
        project_all_buggy = []
        project_all_true = []
        project_all_mix = []

        project_all_buggy_file = open('./sequence_and_label/{}_proj_buggy.txt'.format(project), 'w+',
                                      encoding='utf-8')
        project_all_true_file = open('./sequence_and_label/{}_proj_true.txt'.format(project), 'w+',
                                     encoding='utf-8')
        project_all_mix_file = open('./sequence_and_label/{}_proj_mix.txt'.format(project), 'w+',
                                    encoding='utf-8')

        for version in versions[project]:

            project_version_all_buggy = []
            project_version_all_true = []
            project_version_all_mix = []
            project_version_all_buggy_file = open('./sequence_and_label/{}_{}_ver_buggy.txt'.format(project, version),
                                                  'w+',
                                                  encoding='utf-8')
            project_version_all_true_file = open('./sequence_and_label/{}_{}_ver_true.txt'.format(project, version),
                                                 'w+',
                                                 encoding='utf-8')
            project_version_all_mix_file = open('./sequence_and_label/{}_{}_ver_mix.txt'.format(project, version),
                                                'w+',
                                                encoding='utf-8')

            promise_file_path = "./data/promise_data/{}/{}.csv".format(project, version)
            print(promise_file_path)
            promise_data = csv.reader(open(promise_file_path, 'r'))
            next(promise_data)
            all_count = 0
            exist_count = 0
            sequence_and_label_file = open('./sequence_and_label/{}_{}.txt'.format(project, version), 'w+',
                                           encoding='utf-8')
            corpus_file = open('./corpus/{}_{}.txt'.format(project, version), 'w+', encoding='utf-8')
            for line in promise_data:
                file_name = line[0].replace(".", "/").split("$")[0] + ".java"
                all_count += 1
                for pos_dir in dict_dir[project]:
                    path = "./Code/{}/{}/{}".format(project, version, pos_dir)
                    file_path_file_name = path + "/" + file_name
                    print(file_path_file_name)

                    if os.path.exists(file_path_file_name):
                        # print(str(line[1:]))
                        # get_source(file_path_file_name)
                        # todo token 序列
                        # file_sequence = get_source(file_path_file_name)
                        # todo 节点名字
                        # print(file_name+":  "+file_sequence)
                        file_sequence = parse_ast(file_path_file_name)
                        # corpus_file.write(file_sequence + " ")
                        bugCount = line[-1]
                        # print((bugCount))
                        bugCount = int(bugCount)
                        if bugCount >= 1:
                            bugCount = 1
                            all_buggy.append(str(file_sequence + "\t" + str(bugCount) + "\t") + "\n")
                            project_all_buggy.append(str(file_sequence + "\t" + str(bugCount) + "\t") + "\n")
                            project_version_all_buggy.append(str(file_sequence + "\t" + str(bugCount) + "\t") + "\n")
                        else:
                            all_true.append(str(file_sequence + "\t" + str(bugCount) + "\t") + "\n")
                            project_all_true.append(str(file_sequence + "\t" + str(bugCount) + "\t") + "\n")
                            project_version_all_true.append(str(file_sequence + "\t" + str(bugCount) + "\t") + "\n")
                            bugCount = 0
                        try:
                            all.append(str(file_sequence + "\t" + str(bugCount) + "\t") + "\n")
                            sequence_and_label_file.write(str(file_sequence + "\t" + str(bugCount)) + "\n")
                        # continue
                        except(UnicodeEncodeError):
                            print("UnicodeEncodeError")
            # todo ------------------------------------
            pass
            for item in project_version_all_true:
                try:
                    project_version_all_true_file.write(item)
                except(UnicodeEncodeError):
                    print("UnicodeEncodeError")
            pass
            for item in project_version_all_buggy:
                try:
                    project_version_all_buggy_file.write(item)
                except(UnicodeEncodeError):
                    print("UnicodeEncodeError")
            pass
            project_version_all_mix = get_mix_arr(project_version_all_true, project_version_all_buggy)
            for item in project_version_all_mix:
                try:
                    project_version_all_mix_file.write(item)
                except(UnicodeEncodeError):
                    print("UnicodeEncodeError")
        # todo ------------------------------------
        pass
        for item in project_all_buggy:
            try:
                project_all_buggy_file.write(item)
            except(UnicodeEncodeError):
                print("UnicodeEncodeError")
        pass
        for item in project_all_true:
            try:
                project_all_true_file.write(item)
            except(UnicodeEncodeError):
                print("UnicodeEncodeError")
        pass
        project_all_mix = get_mix_arr(project_all_true, project_all_buggy)
        for item in project_all_mix:
            try:
                project_all_mix_file.write(item)
            except(UnicodeEncodeError):
                print("UnicodeEncodeError")
        pass
    # todo ------------------------------------
    for item in all:
        try:
            all_sequence_and_label_file.write(item)
        except(UnicodeEncodeError):
            print("UnicodeEncodeError")
    pass
    for item in all_buggy:
        try:
            all_buggy_sequence_and_label_file.write(item)
        except(UnicodeEncodeError):
            print("UnicodeEncodeError")
    pass
    for item in all_true:
        try:
            all_true_sequence_and_label_file.write(item)
        except(UnicodeEncodeError):
            print("UnicodeEncodeError")
    pass
    all_mix = get_mix_arr(all_true, all_buggy)
    for item in all_mix:
        try:
            all_mix_sequence_and_label_file.write(item)
        except(UnicodeEncodeError):
            print("UnicodeEncodeError")
