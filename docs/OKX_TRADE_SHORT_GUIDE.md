# OKX 做空（看空）交易规则与下单示例

## 概览

- 做空可通过两类产品实现：`永续合约 (SWAP)`、`交割合约 (FUTURES)`，以及 `币币杠杆 (MARGIN)`。
- 现货 `cash` 模式不支持做空；做空需使用合约或开启杠杆交易（含借币卖出）。
- 下单统一使用 `POST /api/v5/trade/order`。关键参数包括 `instId`、`tdMode`、`side`、`posSide`、`ordType`、`px`、`sz`、`reduceOnly`、`clOrdId` 等。

## 账户与持仓模式

- 账户模式：简单模式 / 单币种保证金 / 跨币种保证金。账户模式需在 Web 端设置。
- 持仓模式：
  - 净持仓 (Net)：仅单边持仓；`posSide` 通常不需要。
  - 双向持仓 (Long/Short)：可同时持有多空；在逐仓且双向模式下，`posSide` 必填（`long` 或 `short`）。
- 杠杆设置：通过 `POST /api/v5/account/set-leverage` 配置，逐仓双向模式需带 `posSide`。

## 下单参数速查

- `instId`：交易标的，例如 `BTC-USDT-SWAP`（永续）、`BTC-USDT-YYYYMMDD`（交割）、`ETH-USDT`（币币/杠杆）。
- `tdMode`：`cash`（现货）、`cross`（全仓）、`isolated`（逐仓）。做空通常为 `cross` 或 `isolated`。
- `side`：`buy`/`sell`。在净持仓模式下，`sell` 用于开空；`buy` 用于平空。
- `posSide`：仅在 **合约** 且 **逐仓 + 双向持仓** 时必填：`long`/`short`。做空时使用 `short`。
- `ordType`：`limit`（限价）/`market`（市价）。建议限价控制风险。
- `px`：限价单价格。
- `sz`：数量（合约为张数或数量；币币为基准币数量）。
- `reduceOnly`：`true` 表示仅减仓，用于确保平空不加仓。
- `clOrdId`：客户端订单 ID（便于追踪）。
- 其他：`expTime`（订单有效截止时间）、`stpMode`（自成交保护）。

## 场景与示例

### 1) 永续合约（净持仓）开空

```json
{
  "instId": "BTC-USDT-SWAP",
  "tdMode": "cross",
  "side": "sell",
  "ordType": "limit",
  "px": "37250",
  "sz": "2",
  "reduceOnly": "false",
  "clOrdId": "short_btc_swap_001"
}
```

- 说明：净持仓模式中，`sell` 即为开空或增加空头敞口；平空用 `buy`。

### 2) 永续合约（逐仓 + 双向持仓）开空

```json
{
  "instId": "BTC-USDT-SWAP",
  "tdMode": "isolated",
  "posSide": "short",
  "side": "sell",
  "ordType": "limit",
  "px": "37250",
  "sz": "2"
}
```

- 说明：`posSide` 选择 `short` 明确开空方向；逐仓需先设置对应杠杆倍数。

### 3) 平空（仅减仓）

```json
{
  "instId": "BTC-USDT-SWAP",
  "tdMode": "isolated",
  "posSide": "short",
  "side": "buy",
  "ordType": "market",
  "reduceOnly": "true",
  "sz": "2"
}
```

- 说明：`reduceOnly` 保证不会反手开多，仅用于减少/关闭空头仓位。

### 4) 币币杠杆做空（借币卖出）

```json
{
  "instId": "ETH-USDT",
  "tdMode": "cross",
  "side": "sell",
  "ordType": "limit",
  "px": "2050",
  "sz": "0.5",
  "clOrdId": "short_eth_margin_001"
}
```

- 说明：需在账户中开启杠杆/借币功能。开空通过卖出基准币实现；后续以 `buy` 方式回补平空。

## 风险与限制

- 限速：REST 私有请求及 WebSocket 登录/订阅均有速率限制；单连接订阅/取消/登录合计约 480 次/小时。
- 自成交保护 (STP)：默认 `Cancel Maker`，可通过下单参数 `stpMode` 控制。
- 有效期：可用 `expTime` 设置请求有效截止时间，超时订单不处理。
- 环境：模拟盘需在请求头携带 `x-simulated-trading: 1`；实盘为 `0`。

## Kokex 系统适配建议

- 现状：`services/trading_tools.py:190` 的 `place_order` 仅传递 `tdMode/side/ordType/sz`，未支持 `posSide`，默认 `tdMode="cash"`。
- 建议：
  - 支持 `tdMode="cross"/"isolated"` 与 `posSide`（合约双向逐仓）。
  - 补充 `reduceOnly`、`stpMode`、`expTime` 等风险控制参数。
  - 合约做空需传入正确 `instId`（如 `BTC-USDT-SWAP`）。
- 参考：`services/trading_tools.py:274` 的 `place_limit_order` 同样需要扩展以支持合约做空参数。

## 参考链接

- 交易与下单：`POST /api/v5/trade/order`（OKX 文档）
- 杠杆与持仓：`POST /api/v5/account/set-leverage`、`GET /api/v5/account/leverage-info`
- 文档入口（中文）：https://www.okx.com/docs-v5/zh/#order-book-trading-trade
- API v5 指南：
  - https://www.okx.com/learn/complete-guide-to-okex-api-v5-upgrade
  - https://web3.okx.com/en-us/learn/complete-guide-to-okex-api-v5-upgrade
