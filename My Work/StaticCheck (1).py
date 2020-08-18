
"""
 * @author NGUYEN LINH DANG MINH // MSSV 1712177
"""
from AST import * 
from Visitor import *
from Utils import Utils
from StaticError import *
from functools import reduce


class MType:
    def __init__(self, partype, rettype):
        self.partype = partype
        self.rettype = rettype

    def __str__(self):
        return 'MType([{}],{})'.format(
            ','.join([str(x) for x in self.partype]),
            str(self.rettype)
        )


class Symbol:
    def __init__(self, name, mtype, value=0):
        self.name = name
        self.mtype = mtype
        self.value = value

    def __str__(self):
        return 'Symbol({},{},{})'.format(
            self.name,
            str(self.mtype),
            str(self.value)
        )


#DEBUG = True
DEBUG = False


def printEnv(env, stop=False):
    if not DEBUG:
        return
    if stop:
        try:
            input(','.join([str(e) for e in env]))
        except EOFError:
            print(','.join([str(e) for e in env]))
    else:
        print(','.join([str(e) for e in env]))


def printDebug(desc, **kwargs):
    if not DEBUG:
        return

    print(desc)

    if 'env' in kwargs:
        if 'stop' in kwargs:
            printEnv(kwargs['env'], kwargs['stop'])
        else:
            printEnv(kwargs['env'])


class StaticChecker(BaseVisitor, Utils):

    global_envi = [
        # from specification, section 7: Built-in Functions/Functions
        Symbol("getInt", MType([], IntType())),
        Symbol("putInt", MType([IntType()], VoidType())),
        Symbol("putIntLn", MType([IntType()], VoidType())),
        Symbol("getFloat", MType([], FloatType())),
        Symbol("putFloat", MType([FloatType()], VoidType())),
        Symbol("putFloatLn", MType([FloatType()], VoidType())),
        Symbol("putBool", MType([BoolType()], VoidType())),
        Symbol("putBoolLn", MType([BoolType()], VoidType())),
        Symbol("putString", MType([StringType()], VoidType())),
        Symbol("putStringLn", MType([StringType()], VoidType())),
        Symbol("putLn", MType([], VoidType()))
    ]

    def __init__(self, ast):
        self.ast = ast

    def check(self):
        return self.visit(self.ast, StaticChecker.global_envi)

    def checkRedeclared(self, symbol, kind, env):
        res = self.lookup(symbol.name.lower(), env, lambda e: e.name.lower())
        #print('tao res nè : ',res,' lại là tao')
        if res is not None:
            raise Redeclared(kind, symbol.name)

    def checkTypeCompatibility(self, lhs, rhs, error):
        # array check
        if isinstance(lhs, ArrayPointerType):
            if not isinstance(rhs, (ArrayType,ArrayPointerType)):
                raise error
            # self.checkTypeCompatibility(lhs.eleType, rhs.eleType, error)
            if not isinstance(lhs.eleType, type(rhs.eleType)):
                raise error

        # float/int coersion
        elif isinstance(lhs, FloatType):
            if not isinstance(rhs, (IntType, FloatType)):
                raise error
        # else
        elif not isinstance(lhs, type(rhs)):
            raise error

    def callBody(self, ast, env):
        '''
        ast: CallStmt | CallExpr
        env: List[Symbol]
        raise TypeMismatchInStatement | TypeMismatchInExpression
        => Type: MType
        ; Used by CallStmt and CallExpr
        ; Both have exact structure difference only on
        ; Raising Error and Kind Expectation
        '''
        
        #print('Tao đang ở Bó Dì nè mày :')
        #print('ddddd:',env)
        #for x in env:
        #    print ('dd:',x)
        #env['kind']=Identifier()
        callParam = [self.visit(x, env) for x in ast.param]
        #print("env=",ast)
        env['kind']=Function()
        mtype = self.visit(  # visits Id
            ast.method,
            env
        )
        env.pop('kind')
        rightParam = mtype.partype

        # lazy init of Error
        error = TypeMismatchInExpression if isinstance(ast, CallExpr) \
            else TypeMismatchInStatement

        if len(rightParam) != len(callParam):
            raise error(ast)

        # LHS’s are formal parameters and RHS’s are arguments
        for pair in zip(rightParam, callParam):
            lhs = pair[0]
            rhs = pair[1]

            self.checkTypeCompatibility(lhs, rhs, error(ast))

        return mtype.rettype

    def loopBody(self, stmts, param):
        '''
        stmt: List[Statement]
        param: {
            'env': List[Symbol],
            'inloop': Bool,
            'rettype': Type
        }
        '''
        env = param['env']
        # for x in param:
        #     print('hihi:',x)
        try:
            rettype = param['rettype']
        except KeyError:
            rettype=''

        outFlag = False

        try:
            for stmt in stmts:
                if outFlag:
                    raise UnreachableStatement(stmt)
                if self.visit(
                    stmt, {
                        'env': env, 'inloop': True, 'rettype': rettype
                    }
                ):
                    outFlag = True
                    #print("NOOOOOO")
        except TypeError:
            #print("NOOOOOOOOOOOOOOOOOOOO")
            self.visit(stmts, {'env': env, 'inloop': True, 'rettype': rettype})

        return False

    def processStatement(self, stmts, param):
        returnFlag = False
        ret_flag=True
        #print('pam=',param)
        for stmt in stmts:
            if returnFlag:
                raise UnreachableStatement(stmt)
            try:
                if (self.visit(stmt,param)==True):
                    #print("boo1")
                    returnFlag = True
            except AttributeError:
                pass
            if (str(stmt).find("Return")==0):
                #print("boo2")
                ret_flag=False   
                returnFlag = False

            if   ("Return" in str(stmt)) and   ("Dowhile" in str(stmt) )   :
                #print("booooooo")
                ret_flag=False   
                returnFlag = False
            #print('stmt=',stmt)
        #returnFlag=True
        return (returnFlag or ret_flag) 

    def visitProgram(self, ast, env):
        printDebug("======SCAN PROGRAM======")
        global_scope = reduce(
            lambda returnList, decl:
                [self.visit(
                    decl,
                    {'env': returnList, 'scan': False}
                )] + returnList,
            ast.decl,
            env[:]
        )
        printDebug("======GLOBAL======", env=global_scope)

        '''if not any(map(
            lambda symbol: isinstance(
                symbol.mtype,
                MType) and symbol.name.lower() == 'main' and isinstance(
                    symbol.mtype.rettype,
                    VoidType) and len(symbol.mtype.partype) == 0,
                global_scope)):
            raise NoEntryPoint()
        '''
        if not any(map(
            lambda symbol: isinstance(
                symbol.mtype,
                MType) and symbol.name.lower() == 'main',
                global_scope)):
            raise NoEntryPoint()

        funcs = filter(lambda x: isinstance(x, FuncDecl), ast.decl)
        for func in funcs:
            self.visit(func, {'env': global_scope, 'scan': True})

        for symbol in global_scope:
            if not isinstance(symbol.mtype, MType):
                continue
            if symbol.name == 'main' :
                continue
            if symbol.value == 0:
                if symbol in env:
                    continue
                #print('why not bitch')
                raise UnreachableFunction(
                    symbol.name)

        return global_scope

    def visitFuncDecl(self, ast, param):
        '''
        ast: FuncDecl
        param: {
            env: List[Symbol], # Global Reference Environment
            scan: Bool
        }
        raise Redeclared(Parameter)
        raise Redeclared(Variable)
        raise UnreachableStatement
        raise FunctionNotReturn
        => Symbol if not scan else None
        '''
        env = param['env']
        scan = param['scan']
        if not scan:
            printDebug("FUNCDECL", env=env, stop=False)
            s = Symbol(
                ast.name.name,
                MType(
                    [x.varType for x in ast.param],
                    ast.returnType
                ))

            kind = Function() if isinstance(ast.returnType, VoidType) \
                else Function()
            self.checkRedeclared(s, kind, env)
            return s
        else:
            printDebug("========SCAN FUNC========")
            printDebug(str(ast))

            try:
                # visits VarDecl -- throws Redeclared(Variable)
                parameter = reduce(
                    lambda scope, vardecl:
                        [self.visit(vardecl, {'env': scope})] + scope,
                    ast.param,
                    # env[:]  # copy
                    []
                )
            except Redeclared as e:
                raise Redeclared(Parameter(), e.n)
            printDebug("PARAM", env=parameter)



                        # check in body
            stmt_list=[]
            local_list=[]
            local_scope=None
            local_scope_x=None
            for x in ast.body.member:
                if not isinstance(x,VarDecl):
                    stmt_list.append(x)
            # for x in stmt_list:
            #     print('xxx=',x)
                if isinstance(x,VarDecl):
                    local_list.append(x)

            # visits VarDecl -- throws Redeclared(Variable)
                local_scope = reduce(
                lambda scope, vardecl:
                    [self.visit(vardecl, {'env': scope})] + scope,
                local_list,
                parameter  # for safety reason, copy
                )

                local_scope_x=local_scope
                for x in env:
                    flag=True
                    #y_x=None
                    for y in local_scope:
                        #print("x.name=",x," y.name=",y)
                        if (x.name==y.name):
                            flag=False
                            y_x=y
                            break
                    if flag:
                        local_scope_x+=[x]
                        #printDebug("local_x true", env=local_scope_x, stop=False)

                    else:
                        flag=True
                        #local_scope_x+=[y_x]
                        #printDebug("local_x false", env=local_scope_x, stop=False)
                self.processStatement(
                    stmt_list,
                    {
                        'env': local_scope_x,
                        'inloop': False,
                        'rettype': ast.returnType,
                        'scan': True,
                        'namae':ast.name.name
                    })
                #local_scope_x+=local_scope
                #for x in local_scope:
                #    print ("x=",x)

            printDebug("LOCAL_VAR", env=local_scope)
            # self.mergeGlobal2Local(local_scope, env)
            
            printDebug("LOCAL_ENV", env=local_scope_x, stop=False)

            # check in body

            # for x in stmt_list:
            #     print('xxx=',x)
            if  self.processStatement(
                    stmt_list,
                    {
                        'env': local_scope_x,
                        'inloop': False,
                        'rettype': ast.returnType,
                        'scan': True,
                        'namae':ast.name.name
                    }) and not isinstance(ast.returnType, VoidType):
                #print("sai rồi nè")
                raise FunctionNotReturn(ast.name.name)

    def visitBlock(self,ast,param):
        env = param['env']
        inloop=param['inloop']
        try:
            scan = param['scan']
        except KeyError:
            #print("Jessie")
            scan=True

        if not scan:
            printDebug("BLOCK DECL nè :", env=env, stop=False)
            return None
        else:
            printDebug("========SCAN member========")
            printDebug(str(ast))

            parameter = []
            printDebug("PARAM", env=parameter)

            local_list=[]
            stmt_list=[]
            local_scope_x=None
            local_scope=None
            for x in ast.member:
                if isinstance(x,VarDecl):
                    local_list.append(x)
                if not isinstance(x,VarDecl):
                    stmt_list.append(x)
            # visits VarDecl -- throws Redeclared(Variable)
                local_scope = reduce(
                    lambda scope, vardecl:
                        [self.visit(vardecl, {'env': scope})] + scope,
                    local_list,
                    parameter  # for safety reason, copy
                )
                local_scope_x=local_scope
                for x in env:
                    flag=True
                    #y_x=None
                    for y in local_scope:
                        #print("x.name=",x," y.name=",y)
                        if (x.name==y.name):
                            flag=False
                            y_x=y
                            break
                    if flag:
                        local_scope_x+=[x]
                        #printDebug("local_x true", env=local_scope_x, stop=False)

                    else:
                        flag=True
                        #local_scope_x+=[y_x]
                        #printDebug("local_x false", env=local_scope_x, stop=False)
                #local_scope_x+=local_scope
                self.processStatement(
                    stmt_list,
                    {
                        'env': local_scope_x,
                        'inloop': inloop,
                        'scan':True,
                        'rettype':param['rettype'],
                        'namae':''

                    })

            printDebug("LOCAL_VAR", env=local_scope)
            # self.mergeGlobal2Local(local_scope, env)
            printDebug("env", env=env, stop=False)

            printDebug("LOCAL_ENV", env=local_scope_x, stop=False)

            # check in body

            self.processStatement(
                    stmt_list,
                    {
                        'env': local_scope_x,
                        'inloop': inloop,
                        'scan':True,
                        'rettype':param['rettype'],
                        'namae':''

                    })
        
    def visitVarDecl(self, ast, param):
        '''
        ast: VarDecl
        param: {
            env: List[Symbol]
            ~~scan: Bool~~ # ignore
        }
        => Symbol
        '''
        # print(param, file=sys.stderr)
        env = param['env']
        printDebug("VARDECL", env=env, stop=False)

        s = Symbol(
            ast.variable,
            ast.varType
        )
        env=set(env)
        env.discard('None')    
        #print("ahihi: ",env)                      
        self.checkRedeclared(s, Variable(), env)

        return s

    def visitIntType(self, asttree, param):
        return None

    def visitFloatType(self, asttree, param):
        return None

    def visitBoolType(self, asttree, param):
        return None

    def visitStringType(self, asttree, param):
        return None

    def visitVoidType(self, asttree, param):
        return None

    def visitArrayType(self, asttree, param):
        return None

    def visitArrayPointerType(self, asttree, param):
        return None

    def visitBinaryOp(self, ast, param):
        '''
        ast: BinaryOp
        param: list[Symbol]
        raise TypeMismatchInExpression
            (/) --> Float/Int:Float/Int => Float
            (+,-,*) --> Float:Float/Int => Float
                    --> Float/Int:Float => Float
                    --> Int:Int         => Int
            (div,mod) --> Int:Int => Int
            (<,<=,=,>=,>,<>) --> Float:Float/Int => Bool
                             --> Float/Int:Float => Bool
                             --> Int:Int => Bool
            (and,or,andthen,orelse) --> Bool:Bool => Bool
        => Type
        '''
        op = ast.op.lower()
        # visits (Id, BinaryOp, UnaryOp, CallExpr, ArrayCell)
        left_type = self.visit(ast.left, param)
        
        right_type = self.visit(ast.right, param)

        def deferType(acceptableTypes, returnType=None):
            if not isinstance(left_type, acceptableTypes):
                raise TypeMismatchInExpression(ast)
            if not isinstance(right_type, acceptableTypes):
                raise TypeMismatchInExpression(ast)


            if returnType is not None:
                return returnType
            if isinstance(left_type, FloatType) or \
                    isinstance(right_type, FloatType):
                return FloatType()
            if isinstance(left_type, type(right_type)):
                return left_type

            raise TypeMismatchInExpression(ast)

        if op in ('!', '&&','||'):
            return deferType((BoolType), BoolType())

        if op in ('%'):
            return deferType((IntType), IntType())

        if op in ('==','!='):
            return deferType((IntType,BoolType), BoolType())

        if op in ('='):
            if ( not isinstance(ast.left,(Id,ArrayCell)) ):
                raise NotLeftValue(ast.left)
            if ( isinstance(left_type,IntType) and not isinstance(right_type,IntType)):
                raise TypeMismatchInExpression(ast)
            if ( isinstance(left_type,(BoolType,StringType))) and not isinstance(right_type,(BoolType,StringType)):
                raise TypeMismatchInExpression(ast)            
            return deferType((IntType,BoolType,FloatType,StringType))

        if op in ('+','-','*','/'):
            return deferType((IntType,FloatType)) 

        if op in ('<','<=','>','>='):
            return deferType((IntType,FloatType), BoolType())


    def visitUnaryOp(self, ast, param):
        '''
        ast: UnaryOp
        param: List[Symbol]
        raise TypeMismatchInExpression
            not Bool => Bool
            - Int => Int
            - Float => Float
        => Type
        '''
        op = ast.op.lower()
        expr = self.visit(ast.body, param)

        if op in ('[]'):
            if not isinstance(expr, (ArrayType,ArrayPointerType) ):
                raise TypeMismatchInExpression(ast)
            return BoolType()

        if op in ('-'):
            if not isinstance(expr, (IntType, FloatType)):
                raise TypeMismatchInExpression(ast)
            return expr

        if op in ('!'):
            if not isinstance(expr, BoolType):
                raise TypeMismatchInExpression(ast)
            return BoolType()

    def visitCallExpr(self, ast, env):
        '''
        ast: CallExpr ~ CallStmt
        env: list[Symbol]
        raise Undeclared(Function)
        raise TypeMismatchInExpression
            wrong param size
            wrong param type
                Array[n..m] of X --> Array[n..m] of X
                Float --> Float/Int
                X --> X
        => Type
        '''
        return self.callBody(ast, env)

    def visitId(self, ast, param):
        '''
        ast: Id
        param: List[Symbol] or {
            'env': List[Symbol]
            'kind': kind expectation
        }
        raise Undeclared
        => Type: Symbol.mtype
        '''
        #print('param=',type(param))
        env = param['env'] if not isinstance(param, list) else param
        #print('param=',env)

        try:
            kind = param['kind'] if not isinstance(param, list) else Identifier()
        except KeyError:
            kind=Identifier()

        # type(res) == Symbol
        # printDebug("ID", env=env, stop=False)

        #print(env['env'])

        # for x in env['env']:
        #     print('x=',x)
        #     ls.append(x)
        
        try:
            res = self.lookup(ast.name.lower(), env, lambda e: e.name.lower())
        except AttributeError:
            res = self.lookup(ast.name.lower(),env, lambda e: e.lower())

        #res="stupid python"

        #print("res=",res,' kind=',kind)

        # for x in env:
        #     print(x,'=',env[x])
        # for x in env['env']:
        #     print('envxxx=',x)
            
        if res is None:
            
            raise Undeclared(kind, ast.name)

        if isinstance(kind, Identifier):
            if isinstance(res.mtype, MType):
                raise Undeclared(kind, ast.name)
            return res.mtype

        # param is dict
        if isinstance(kind, Function):
            # check if mtype -- aka function
            if not isinstance(res.mtype, MType):
                raise Undeclared(kind, ast.name)

            #print('name:',param['namae'],' ast.name:',ast.name)

            if (param['namae']!=ast.name):
                res.value += 1

            #print(res.value)

            return res.mtype

    def visitArrayCell(self, ast, param):
        '''
        ast: ArrayCell
        param: List[Symbol]
        raise TypeMismatchInExpression
            arr[idx] --> ArrayType[IntType] => ArrayType.eleType
        => Type
        '''
        arr = self.visit(ast.arr, param)
        idx = self.visit(ast.idx, param)

        if not isinstance(idx, IntType) or \
                not isinstance(arr, (ArrayType,ArrayPointerType)):
            raise TypeMismatchInExpression(ast)

        return arr.eleType

    '''
    Statements are passed with a dict() with simple params:
        env: List[Symbol]   # local referencing environment
        inloop: Bool        # in loop flag
        rettype: Type       # return type of function/Function
    Simple statements deal only with 'env',
    Continue/Break statements use 'inloop' to check for non in loop call
    Return statements use 'rettype' to check for type compatibility when return
    Function/Function statements pass all these params,
    For/While statements pass inloop as True when processing loop statements
                while preserving 'rettype
    If statements pass param to then/else statements
    With statements ...
    All other statements uses param as read only
    Return of statements are the status of the function/Function whether
        it has returned or not
    For example:
    Function main();
    begin
        if (n and mask) then
            return 1;
        else
            return 0;
    end
    the corresponding AST Tree will be:
    Program([
        FuncDecl(Id(main),[],VoidType(),[],[
            If(BinaryOp(and,Id(n),Id(mask)),[
                Return(Some(IntLiteral(1)))
            ],[
                Return(Some(IntLiteral(0)))
            ])
        ])
    ])
    The If in AST will return True as both of the branch returns.
    There is no else if statement, but this algorithm works as
    expect that in if statement, all branch returns means the function is ended
    The same logic can also be applied to for/while/with statements
    '''

    def visitIf(self, ast, param):
        '''
        ast: If
        param: {
            'env': List[Symbol]
            'inloop': Bool
            'rettype': Type
        }
        => Returned?
        '''
        expr = self.visit(ast.expr, param['env'])
        if not isinstance(expr, BoolType):
            raise TypeMismatchInStatement(ast)

        self.visit(ast.thenStmt,param)

        try:
            self.visit(ast.elseStmt,param)
        except AttributeError:
            pass
        return False

    def visitFor(self, ast, param):
        '''
        ast: For
        param: {
            'env': List[Symbol],
            'rettype': Type
        }
        '''

        env = param
        #printDebug("ENV for ", env=env)
        expr1 = self.visit(ast.expr1, env)
        expr2 = self.visit(ast.expr2, env)
        expr3 = self.visit(ast.expr3, env)
    
        if not isinstance(expr1, IntType) or \
                not isinstance(expr3, IntType) or \
                not isinstance(expr2, BoolType):
            raise TypeMismatchInStatement(ast)

        return self.loopBody(ast.loop, param)

    def visitContinue(self, ast, param):
        '''
        ast: Continue
        param: {
            'inloop': Bool
        }
        '''
        if not param['inloop']:
            raise ContinueNotInLoop()
        return True

    def visitBreak(self, ast, param):
        '''
        ast: Break
        param: {
            'inloop': Bool
        }
        '''
        if not param['inloop']:
            raise BreakNotInLoop()
        return True

    def visitReturn(self, ast, param):
        '''
        ast: Return
        param: {
            'env': List[Symbol]
            'rettype': Type
        }
        '''
        rettype = param['rettype']
        env = param
        if isinstance(rettype, VoidType):
            if ast.expr is not None:
                raise TypeMismatchInStatement(ast)
        else:
            if ast.expr is None:
                raise TypeMismatchInStatement(ast)
            self.checkTypeCompatibility(
                rettype, self.visit(ast.expr, env),
                TypeMismatchInStatement(ast)
            )

        return True

    def visitDowhile(self, ast, param):
        '''
        ast: While
        param: {
            'env': List[Symbol],
            'inloop': Bool,
            'rettype': Type
        }
        '''
        env = param
        env['namae']=''
        exp = self.visit(ast.exp, env)
        # for x in param:
        #     print('x=',x)
        if not isinstance(exp, BoolType):
            raise TypeMismatchInStatement(ast)

        return self.loopBody(ast.sl, param)

    def visitIntLiteral(self, asttree, param):
        #print("phải hông")
        return IntType()

    def visitFloatLiteral(self, asttree, param):
        return FloatType()

    def visitBooleanLiteral(self, asttree, param):
        return BoolType()

    def visitStringLiteral(self, asttree, param):
        return StringType()