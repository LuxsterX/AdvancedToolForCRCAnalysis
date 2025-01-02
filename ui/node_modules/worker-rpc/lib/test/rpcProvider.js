"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var assert_1 = __importDefault(require("assert"));
var RpcProvider_1 = __importDefault(require("../src/RpcProvider"));
suite('RPC provider', function () {
    var local, remote, transferLocalToRemote, transferRemoteToLocal, errorLocal, errorRemote;
    setup(function () {
        local = new RpcProvider_1.default(function (message, transfer) { return (transferLocalToRemote = transfer, remote.dispatch(message)); }, 50);
        local.error.addHandler(function (err) { return errorLocal = err; });
        remote = new RpcProvider_1.default(function (message, transfer) { return (transferRemoteToLocal = transfer, local.dispatch(message)); }, 50);
        remote.error.addHandler(function (err) { return errorRemote = err; });
        transferLocalToRemote = transferRemoteToLocal = undefined;
        errorRemote = errorLocal = undefined;
    });
    suite('signals', function () {
        test('Signals are propagated', function () {
            var x = -1;
            remote.registerSignalHandler('action', function (value) { return x = value; });
            local.signal('action', 5);
            assert_1.default(!errorLocal);
            assert_1.default(!errorRemote);
            assert_1.default.strictEqual(x, 5);
        });
        test('Unregistered signals raise an error', function () {
            local.signal('action', 10);
            assert_1.default(errorLocal);
            assert_1.default(errorRemote);
        });
        test('Multiple signals do not interfere', function () {
            var x = -1, y = -1;
            remote.registerSignalHandler('setx', function (value) { return x = value; });
            remote.registerSignalHandler('sety', function (value) { return y = value; });
            local.signal('setx', 5);
            local.signal('sety', 6);
            assert_1.default(!errorLocal);
            assert_1.default(!errorRemote);
            assert_1.default.strictEqual(x, 5);
            assert_1.default.strictEqual(y, 6);
        });
        test('Multiple handlers can be bound to one signal', function () {
            var x = -1;
            remote.registerSignalHandler('action', function (value) { return x = value; });
            local.signal('action', 1);
            local.signal('action', 2);
            assert_1.default(!errorLocal);
            assert_1.default(!errorRemote);
            assert_1.default.strictEqual(x, 2);
        });
        test('Handlers can be deregistered', function () {
            var x = -1;
            var handler = function (value) { return x = value; };
            remote.registerSignalHandler('action', handler);
            remote.deregisterSignalHandler('action', handler);
            local.signal('action', 5);
            assert_1.default(!errorLocal);
            assert_1.default(!errorRemote);
            assert_1.default.strictEqual(x, -1);
        });
        test('Transfer is honored', function () {
            var x = -1;
            var transfer = [1, 2, 3];
            remote.registerSignalHandler('action', function (value) { return x = value; });
            local.signal('action', 2, transfer);
            assert_1.default(!errorLocal);
            assert_1.default(!errorRemote);
            assert_1.default.strictEqual(x, 2);
            assert_1.default.strictEqual(transferLocalToRemote, transfer);
            assert_1.default(!transferRemoteToLocal);
        });
    });
    suite('RPC', function () {
        test('RPC handlers can return values', function () {
            remote.registerRpcHandler('action', function () { return 10; });
            return local
                .rpc('action')
                .then(function (result) { return (assert_1.default.strictEqual(result, 10),
                assert_1.default(!errorLocal),
                assert_1.default(!errorRemote)); });
        });
        test('RPC handlers can return promises', function () {
            remote.registerRpcHandler('action', function () { return new Promise(function (r) { return setTimeout(function () { return r(10); }, 15); }); });
            return local
                .rpc('action')
                .then(function (result) { return (assert_1.default.strictEqual(result, 10),
                assert_1.default(!errorLocal),
                assert_1.default(!errorRemote)); });
        });
        test('Promise rejection is transferred', function () {
            remote.registerRpcHandler('action', function () { return new Promise(function (resolve, reject) { return setTimeout(function () { return reject(10); }, 15); }); });
            return local
                .rpc('action')
                .then(function () { return Promise.reject('should have been rejected'); }, function (result) { return (assert_1.default.strictEqual(result, 10),
                assert_1.default(!errorLocal),
                assert_1.default(!errorRemote)); });
        });
        test('Invalid RPC calls are rejected', function () {
            return local
                .rpc('action')
                .then(function () { return Promise.reject('should have been rejected'); }, function () { return undefined; });
        });
        test('Invalid RPC calls throw on both ends', function () {
            return local
                .rpc('action')
                .then(function () { return Promise.reject('should have been rejected'); }, function () { return undefined; })
                .then(function () { return (assert_1.default(errorLocal),
                assert_1.default(errorRemote)); });
        });
        test('RPC calls time out', function () {
            remote.registerRpcHandler('action', function () { return new Promise(function (r) { return setTimeout(function () { return r(10); }, 100); }); });
            return local
                .rpc('action')
                .then(function () { return Promise.reject('should have been rejected'); }, function () { return (assert_1.default(errorLocal), new Promise(function (r) { return setTimeout(r, 100); })); })
                .then(function () { return assert_1.default(errorRemote); });
        });
        test('Multiple RPC handlers do not interfere', function () {
            remote.registerRpcHandler('a1', function (value) { return new Promise(function (r) { return setTimeout(function () { return r(value); }, 30); }); });
            remote.registerRpcHandler('a2', function (value) { return 2 * value; });
            return Promise
                .all([
                local.rpc('a1', 10),
                local.rpc('a2', 20)
            ])
                .then(function (_a) {
                var r1 = _a[0], r2 = _a[1];
                return (assert_1.default.strictEqual(r1, 10),
                    assert_1.default.strictEqual(r2, 40),
                    assert_1.default(!errorLocal),
                    assert_1.default(!errorRemote));
            });
        });
        test('RPC handler can be deregistered', function () {
            var handler = function () { return 10; };
            remote.registerRpcHandler('action', handler);
            remote.deregisterRpcHandler('action', handler);
            return local
                .rpc('action')
                .then(function () { return Promise.reject('should have been rejected'); }, function () { return (assert_1.default(errorLocal),
                assert_1.default(errorRemote)); });
        });
        test('Transfer is honored', function () {
            var transfer = [1, 2, 3];
            remote.registerRpcHandler('action', function () { return 10; });
            return local
                .rpc('action', undefined, transfer)
                .then(function (x) { return (assert_1.default.strictEqual(transferLocalToRemote, transfer),
                assert_1.default.strictEqual(x, 10),
                assert_1.default(!errorLocal),
                assert_1.default(!errorRemote)); });
        });
    });
});
