import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import DigitalHumanSetting from '../index'

const getOpenClawDetectedConfigMock = vi.fn()

vi.mock('@/apis/dip-studio/guide', () => ({
  getOpenClawDetectedConfig: () => getOpenClawDetectedConfigMock(),
}))

vi.mock('../digitalHumanStore', () => ({
  useDigitalHumanStore: (selector: (state: { uiMode: string }) => unknown) =>
    selector({ uiMode: 'view' }),
}))

vi.mock('../BasicSetting', () => ({
  default: () => <div>basic-setting</div>,
}))

vi.mock('../SkillConfig', () => ({
  default: () => <div>skill-config</div>,
}))

vi.mock('../KnowledgeConfig', () => ({
  default: () => <div>knowledge-config</div>,
}))

vi.mock('../ChannelConfig', () => ({
  default: () => <div>channel-config</div>,
}))

vi.mock('@/components/IconFont', () => ({
  default: () => <span data-testid="icon-font" />,
}))

describe('DigitalHumanSetting', () => {
  it('配置了 KWeaver 服务地址时显示知识配置菜单', async () => {
    getOpenClawDetectedConfigMock.mockResolvedValueOnce({
      openclaw_address: 'ws://127.0.0.1:3000',
      openclaw_token: 'token',
      kweaver_base_url: 'https://kweaver.example.com',
    })

    render(<DigitalHumanSetting />)

    await waitFor(() => {
      expect(
        screen.getByRole('button', { name: 'digitalHuman.setting.menuKnowledge' }),
      ).toBeInTheDocument()
    })
  })

  it('未配置 KWeaver 服务地址时隐藏知识配置菜单', async () => {
    getOpenClawDetectedConfigMock.mockResolvedValueOnce({
      openclaw_address: 'ws://127.0.0.1:3000',
      openclaw_token: 'token',
      kweaver_base_url: '',
    })

    render(<DigitalHumanSetting />)

    await waitFor(() => {
      expect(
        screen.queryByRole('button', { name: 'digitalHuman.setting.menuKnowledge' }),
      ).not.toBeInTheDocument()
    })
  })
})
