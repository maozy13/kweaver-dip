import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import PreviewArtifact from '..'
import { getSessionArchiveSubpath } from '../../../../apis'
import type { DipChatKitPreviewPayload } from '../../../../types'

vi.mock('../../../../apis', () => ({
  getSessionArchiveSubpath: vi.fn(),
}))

vi.mock('react-intl-universal', () => ({
  default: {
    get: (_key: string) => ({
      d: (fallback: string) => fallback,
    }),
  },
}))

const mockedGetSessionArchiveSubpath = vi.mocked(getSessionArchiveSubpath)

const baseProps = {
  onClose: vi.fn(),
  fullscreen: false,
  onToggleFullscreen: vi.fn(),
}

const renderPreviewArtifact = (payload: DipChatKitPreviewPayload) =>
  render(<PreviewArtifact {...baseProps} payload={payload} />)

afterEach(() => {
  vi.clearAllMocks()
})

describe('PreviewArtifact', () => {
  it('loads directory entries via json and keeps the list visible while previewing a file', async () => {
    mockedGetSessionArchiveSubpath.mockResolvedValueOnce({
      path: '2026-03-25-03-04-05/output',
      contents: [
        { name: 'reports', type: 'directory' },
        { name: 'summary.md', type: 'file' },
      ],
    })
    mockedGetSessionArchiveSubpath.mockResolvedValueOnce('# summary')

    renderPreviewArtifact({
      title: '目录预览：output',
      content: '2026-03-25-03-04-05/output',
      sourceType: 'artifact',
      artifact: {
        sessionKey: 'session-1',
        subpath: '2026-03-25-03-04-05/output',
        fileName: 'output',
        archiveRoot: 'archives/chat-1',
        entryType: 'directory',
      },
    })

    await waitFor(() => {
      expect(mockedGetSessionArchiveSubpath).toHaveBeenCalledWith(
        'session-1',
        '2026-03-25-03-04-05/output',
        { responseType: 'json' },
      )
    })

    expect(await screen.findByText('reports')).toBeInTheDocument()
    expect(screen.getAllByText('summary.md')).not.toHaveLength(0)

    await waitFor(() => {
      expect(mockedGetSessionArchiveSubpath).toHaveBeenLastCalledWith(
        'session-1',
        '2026-03-25-03-04-05/output/summary.md',
        { responseType: 'text' },
      )
    })

    expect(await screen.findByRole('heading', { name: 'summary' })).toBeInTheDocument()
    expect(screen.getByText('reports')).toBeInTheDocument()
    expect(screen.getAllByText('summary.md')).not.toHaveLength(0)
  })

  it('hides directory list and close button in fullscreen mode', async () => {
    mockedGetSessionArchiveSubpath.mockResolvedValueOnce({
      path: '2026-03-25-03-04-05/output',
      contents: [
        { name: 'reports', type: 'directory' },
        { name: 'summary.md', type: 'file' },
      ],
    })
    mockedGetSessionArchiveSubpath.mockResolvedValueOnce('# summary')

    render(
      <PreviewArtifact
        {...baseProps}
        fullscreen
        payload={{
          title: '目录预览：output',
          content: '2026-03-25-03-04-05/output',
          sourceType: 'artifact',
          artifact: {
            sessionKey: 'session-1',
            subpath: '2026-03-25-03-04-05/output',
            fileName: 'output',
            archiveRoot: 'archives/chat-1',
            entryType: 'directory',
          },
        }}
      />,
    )

    expect(await screen.findByRole('heading', { name: 'summary' })).toBeInTheDocument()
    expect(screen.queryByText('reports')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('关闭预览')).not.toBeInTheDocument()
    expect(screen.getByLabelText('退出全屏')).toBeInTheDocument()
    expect(screen.getByLabelText('下载文件')).toBeInTheDocument()
  })

  it('hides header while fullscreen preview scrolls down and shows it again at top', async () => {
    mockedGetSessionArchiveSubpath.mockResolvedValueOnce('# summary')

    const { container } = render(
      <PreviewArtifact
        {...baseProps}
        fullscreen
        payload={{
          title: 'summary.md',
          content: '2026-03-25-03-04-05/output/summary.md',
          sourceType: 'artifact',
          artifact: {
            sessionKey: 'session-1',
            subpath: '2026-03-25-03-04-05/output/summary.md',
            fileName: 'summary.md',
            archiveRoot: 'archives/chat-1',
            entryType: 'file',
          },
        }}
      />,
    )

    await waitFor(() => {
      expect(mockedGetSessionArchiveSubpath).toHaveBeenCalledWith(
        'session-1',
        '2026-03-25-03-04-05/output/summary.md',
        { responseType: 'text' },
      )
    })

    const header = container.querySelector('[data-header-hidden]') as HTMLElement
    const scrollArea = container.querySelector('.ScrollContainer > div') as HTMLDivElement

    expect(header.dataset.headerHidden).toBe('false')

    Object.defineProperty(scrollArea, 'scrollTop', {
      configurable: true,
      value: 120,
      writable: true,
    })
    fireEvent.scroll(scrollArea)

    expect(header.dataset.headerHidden).toBe('true')

    Object.defineProperty(scrollArea, 'scrollTop', {
      configurable: true,
      value: 0,
      writable: true,
    })
    fireEvent.scroll(scrollArea)

    expect(header.dataset.headerHidden).toBe('false')
  })

  it('exits fullscreen when pressing Escape', async () => {
    mockedGetSessionArchiveSubpath.mockResolvedValueOnce('# summary')
    const onToggleFullscreen = vi.fn()

    render(
      <PreviewArtifact
        {...baseProps}
        fullscreen
        onToggleFullscreen={onToggleFullscreen}
        payload={{
          title: 'summary.md',
          content: '2026-03-25-03-04-05/output/summary.md',
          sourceType: 'artifact',
          artifact: {
            sessionKey: 'session-1',
            subpath: '2026-03-25-03-04-05/output/summary.md',
            fileName: 'summary.md',
            archiveRoot: 'archives/chat-1',
            entryType: 'file',
          },
        }}
      />,
    )

    await waitFor(() => {
      expect(mockedGetSessionArchiveSubpath).toHaveBeenCalledWith(
        'session-1',
        '2026-03-25-03-04-05/output/summary.md',
        { responseType: 'text' },
      )
    })

    fireEvent.keyDown(window, { key: 'Escape' })

    expect(onToggleFullscreen).toHaveBeenCalledTimes(1)
  })
})
