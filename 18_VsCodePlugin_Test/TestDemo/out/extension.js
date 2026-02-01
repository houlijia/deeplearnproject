"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const http = __importStar(require("http"));
const url = __importStar(require("url"));
let server = null;
const SERVER_PORT = 3000;
function activate(context) {
    console.log('Congratulations, your extension "my-http-api-extension" is now active!');
    // 注册一个简单的命令
    let disposable = vscode.commands.registerCommand('TestDemo.HelloWorld', () => {
        vscode.window.showInformationMessage('Hello World from TestDemo!');
    });
    context.subscriptions.push(disposable);
    startHttpServer(context);
}
function startHttpServer(context) {
    server = http.createServer((req, res) => {
        const parsedUrl = url.parse(req.url, true);
        const path = parsedUrl.pathname;
        res.setHeader('Access-Control-Allow-Origin', '*');
        res.setHeader('Access-Control-Request-Method', '*');
        res.setHeader('Access-Control-Allow-Methods', 'OPTIONS, GET, POST');
        res.setHeader('Access-Control-Allow-Headers', '*');
        if (req.method === 'OPTIONS') {
            res.writeHead(200);
            res.end();
            return;
        }
        if (req.method === 'POST' && path === '/HelloWorld') {
            let body = '';
            try {
                // console.log('Received request body:', body); // 用于调试
                // 尝试获取当前活动的编辑器上下文
                // 注意：executeCommand 应该在 VSCode 的主循环中执行，这通常由用户交互或扩展内部调度保证。
                // 直接在 HTTP 回调中调用通常是安全的，因为 VSCode 的事件循环会处理它。
                // 但我们仍需检查是否存在必要的上下文 (e.g., activeTextEditor)
                const editor = vscode.window.activeTextEditor;
                if (!editor) {
                    throw new Error('No active text editor available. Please open a file and ensure it has focus.');
                }
                // 执行命令
                // 使用 executeCommand 返回一个 Promise，我们可以等待它完成
                let aa = vscode.commands.executeCommand('TestDemo.HelloWorld');
                console.log(aa);
                // 成功响应
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    success: true,
                    message: 'Hello World command executed successfully!' + aa
                }));
            }
            catch (error) {
                console.error('Error executing command via HTTP API:', error);
                // 错误响应
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    success: false,
                    error: error.message || 'An unknown error occurred while executing the command.'
                }));
            }
        }
        else if (req.method === 'GET' && path === '/status') {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({
                status: 'OK',
                message: 'VSCode Extension API is running on port ' + SERVER_PORT,
                timestamp: new Date().toISOString()
            }));
        }
        else {
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: false, error: 'Route not found' }));
        }
    });
    server.listen(SERVER_PORT, 'localhost', () => {
        console.log(`HTTP server running at http://localhost:${SERVER_PORT}/`);
        vscode.window.showInformationMessage(`Extension API server started on port ${SERVER_PORT}`);
    });
    context.subscriptions.push({
        dispose: () => {
            if (server) {
                server.close(() => {
                    console.log('HTTP server closed.');
                });
                server = null;
            }
        }
    });
}
function deactivate() {
    console.log('Extension is deactivating...');
    // 清理由 context.subscriptions 自动处理
}
// import * as vscode from 'vscode';
// export function activate(context: vscode.ExtensionContext) {
// 	console.log('Congratulations, your extension "TestDemo" is now active!');
// 	const disposable = vscode.commands.registerCommand('TestDemo.helloWorld', () => {
// 		vscode.window.showInformationMessage('Hello World from TestDemo!');
// 	});
// 	context.subscriptions.push(disposable);
// }
// export function deactivate() {}
//# sourceMappingURL=extension.js.map